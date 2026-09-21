"""Agent loop orchestrating client calls, tool use and message history."""

import sys
import time
from collections.abc import Iterator
from itertools import count
from typing import Dict, List

from arancio.core.builders.system_prompt import SystemPromptBuilder
from arancio.core.clients.base import BaseClient
from arancio.core.constants.agent import (
    AGENT_DEFAULT_MAX_RETRIES,
    AGENT_DEFAULT_MAX_TURNS,
    AGENT_DEFAULT_TURN_WAIT_TIME,
    AGENT_DEFAULT_TURN_WAIT_TIME_MULTIPLIER,
    AGENT_UNLIMITED_MAX_TURNS,
)
from arancio.core.hooks.manager import HookManager
from arancio.core.hooks.types import Hook
from arancio.core.messages import (
    ChunkMessage,
    ErrorMessage,
    Message,
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
    UserMessage,
)
from arancio.core.permissions.manager import PermissionManager
from arancio.core.permissions.types import (
    PermissionCategory,
    PermissionDecision,
    PermissionLevel,
    PermissionOutcome,
)
from arancio.core.tools.base import BaseTool
from arancio.core.tools.schema import ToolSchema


class Agent:
    """Coding agent driving a multi-turn loop against an LLM client."""

    def __init__(
        self,
        client: BaseClient,
        permission_manager: PermissionManager,
        hook_manager: HookManager,
        message_history: List[Message] | None = None,
        max_turns: int | str = AGENT_DEFAULT_MAX_TURNS,
        max_retries: int = AGENT_DEFAULT_MAX_RETRIES,
        retry_delay: float = AGENT_DEFAULT_TURN_WAIT_TIME,
        retry_delay_multiplier: float = AGENT_DEFAULT_TURN_WAIT_TIME_MULTIPLIER,
    ) -> None:
        """Initialize the agent with a client, permission manager and hook manager.

        Args:
            client: the LLM client used to send requests.
            permission_manager: the permission manager that creates the agent's
                tools and gates each tool call.
            hook_manager: the hook manager the loop and system-prompt build
                dispatch through (see AGENTS.md's Hooks section).
            message_history: the model context the agent starts from, copied so
                the caller keeps no handle on it. Defaults to none, an empty
                history; a restored session passes its saved messages here.
            max_turns: the maximum number of loop turns before aborting, or
                ``"inf"`` (the default) for no limit: the loop runs until the
                model stops requesting tools.
            max_retries: the maximum number of consecutive failed turns before
                the run aborts.
            retry_delay: seconds to wait before retrying a failed turn, so the
                turn is not retried immediately.
            retry_delay_multiplier: factor the retry wait grows by on each
                consecutive failed-turn retry (exponential backoff).
        """
        self._client: BaseClient = client
        self._max_turns: int | str = max_turns
        self._max_retries: int = max_retries
        self._retry_delay: float = retry_delay
        self._retry_delay_multiplier: float = retry_delay_multiplier
        self._message_history: List[Message] = list(message_history or [])
        self._system_prompt_builder: SystemPromptBuilder = SystemPromptBuilder()
        self._permission_manager: PermissionManager = permission_manager
        self._hook_manager: HookManager = hook_manager

        self._tools: Dict[str, BaseTool] = {}
        self._refresh_tools()

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the agent.

        Returns:
            A compact string with configuration and state counts.
        """
        return (
            f"{type(self).__name__}("
            f"client={self._client!r}, "
            f"permissions={self._permission_manager!r}, "
            f"max_turns={self._max_turns!r}, "
            f"max_retries={self._max_retries!r}, "
            f"tools={list(self._tools.values())!r}"
            ")"
        )

    def __str__(self) -> str:
        """Return a pretty, multi-line representation of the agent.

        Returns:
            An indented, one-tool-per-line rendering suitable for printing.
        """
        if self._tools:
            tools = "\n".join(f"        {tool!r}," for tool in self._tools.values())
            tools_block = f"[\n{tools}\n    ]"
        else:
            tools_block = "[]"
        return (
            f"{type(self).__name__}(\n"
            f"    client={self._client!r},\n"
            f"    permissions={self._permission_manager!r},\n"
            f"    max_turns={self._max_turns!r},\n"
            f"    max_retries={self._max_retries!r},\n"
            f"    tools={tools_block},\n"
            ")"
        )

    @property
    def _tool_schemas(self) -> List[ToolSchema]:
        return [tool.schema() for tool in self._tools.values()]

    @property
    def max_turns(self) -> int | str:
        """Return the maximum number of loop turns per run.

        Returns:
            The maximum number of turns before the run aborts, or ``"inf"``
            when the run is unlimited.
        """
        return self._max_turns

    @max_turns.setter
    def max_turns(self, value: int | str) -> None:
        """Set the maximum number of loop turns per run.

        Args:
            value: the new maximum number of turns, or ``"inf"`` for no limit.
        """
        self._max_turns = value

    @property
    def max_retries(self) -> int:
        """Return the maximum number of consecutive failed turns per run.

        Returns:
            The maximum number of consecutive failed turns before the run
            aborts.
        """
        return self._max_retries

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        """Set the maximum number of consecutive failed turns per run.

        Args:
            value: the new maximum number of consecutive failed turns.
        """
        self._max_retries = value

    @property
    def retry_delay(self) -> float:
        """Return the base wait before retrying a failed turn.

        Returns:
            The base retry wait in seconds.
        """
        return self._retry_delay

    @retry_delay.setter
    def retry_delay(self, value: float) -> None:
        """Set the base wait before retrying a failed turn.

        Args:
            value: the new base retry wait in seconds.
        """
        self._retry_delay = value

    @property
    def retry_delay_multiplier(self) -> float:
        """Return the factor the retry wait grows by on each consecutive retry.

        Returns:
            The exponential-backoff multiplier.
        """
        return self._retry_delay_multiplier

    @retry_delay_multiplier.setter
    def retry_delay_multiplier(self, value: float) -> None:
        """Set the factor the retry wait grows by on each consecutive retry.

        Args:
            value: the new exponential-backoff multiplier.
        """
        self._retry_delay_multiplier = value

    def add_tool(self, tool: BaseTool):
        """Append a tool to the agent's tool catalog.

        Args:
            tool: the tool definition to expose to the model.
        """
        self._tools[tool.name] = tool

    def _refresh_tools(self) -> None:
        """Rebuild the tool catalog from the permission manager's grants."""
        self._tools = {
            tool.name: tool for tool in self._permission_manager.allowed_tools
        }

    def add_permission(
        self,
        category: PermissionCategory,
        level: PermissionLevel = PermissionLevel.ASK,
    ) -> None:
        """Grant a category and rebuild the tool catalog.

        Args:
            category: the category to grant.
            level: the permission level. Defaults to
                :attr:`PermissionLevel.ASK`.
        """
        self._permission_manager.add_permission(category, level)
        self._refresh_tools()

    def remove_permission(self, category: PermissionCategory) -> None:
        """Revoke a category's grant and rebuild the tool catalog.

        Args:
            category: the category whose grant is removed.
        """
        self._permission_manager.remove_permission(category)
        self._refresh_tools()

    def set_permission_level(
        self, category: PermissionCategory, level: PermissionLevel
    ) -> None:
        """Change a category's permission level and rebuild the tool catalog.

        Args:
            category: the category whose grant to update.
            level: the new permission level.
        """
        self._permission_manager.set_permission_level(category, level)
        self._refresh_tools()

    def set_permissions(
        self, permissions: dict[PermissionCategory, PermissionLevel]
    ) -> None:
        """Replace all permission grants and rebuild the tool catalog.

        Args:
            permissions: the new category-to-level grants, replacing the
                current ones wholesale.
        """
        self._permission_manager.set_permissions(permissions)
        self._refresh_tools()

    def clear_history(self) -> None:
        """Empty the conversation history, as if starting a new chat."""
        self._message_history = []

    def add_message_to_history(self, message: Message) -> None:
        """Append a single message to the conversation history.

        Args:
            message: the message to add.
        """
        self._message_history.append(message)

    @staticmethod
    def _permission_message(
        call: ToolCallMessage, decision: PermissionDecision
    ) -> Message | None:
        """Build the model-facing message a resolved permission decision needs.

        Args:
            call: the tool call the decision resolves.
            decision: the permission decision to word for the model.

        Returns:
            The note wrapped in a report-then-answer instruction when the call
            is allowed with one, the tool error describing the denial or the
            missing tool when it is refused, and ``None`` for a plain allow.
        """
        if decision.outcome is PermissionOutcome.ALLOWED:
            if not decision.note:
                return None

            # the note instructs the model to report the result first, then
            # answer it, so it is sent after the result
            instruction = (
                f"While running tool {call.id}: {call.name}, user also noted: "
                f"{decision.note}. First report tool calling result, then "
                f"answer user note."
            )
            return UserMessage(content=instruction, display_text=decision.note)

        # a missing tool must not read as a user denial: the model asked for a
        # tool that is not part of this run at all
        if decision.outcome is PermissionOutcome.UNAVAILABLE:
            return ToolErrorMessage(
                content=f"Tool {call.name} does not exist.", id=call.id
            )

        content = f"Tool call {call.name} denied by user."
        if decision.note:
            content += f" Additional information from user: {decision.note}"
        return ToolErrorMessage(content=content, id=call.id)

    def _emit(self, message: Message) -> Iterator[Message]:
        """Append a message to history and emit it.

        Args:
            message: the finalized message to append and emit.

        Yields:
            The message itself.
        """
        self.add_message_to_history(message)
        yield message

    def restore_history(self, messages: List[Message]) -> None:
        """Replace model history with messages restored from a session.

        Args:
            messages: the restored model-context messages in original order.
        """
        self._message_history = list(messages)

    def _run_tool(
        self, call: ToolCallMessage
    ) -> tuple[ToolResultMessage, list[Message]]:
        """Execute a tool call and return its result alongside hook messages.

        Args:
            call: the tool call message to execute.

        Returns:
            The tool result and any hook messages.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            message = f"Unknown tool {call.name}"
            return ToolErrorMessage(
                content=message,
                id=call.id,
            ), []
        else:
            return tool.call(call_id=call.id, **(call.arguments or {}))

    def __call__(
        self, message: Message, prelude: List[Message] | None = None
    ) -> Iterator[Message]:
        """Run the agent loop starting from the given user message.

        The user message, every parsed client message, and every local
        tool result are appended to conversation history. Produced client
        and tool-result messages are yielded in order as they are handled.
        The loop ends naturally when the model replies without tool calls;
        when ``max_turns`` is an integer, it also aborts after that many
        turns, and it always aborts after ``max_retries`` consecutive failed
        turns.

        Dispatches hooks through this agent's hook manager as the run
        progresses (see AGENTS.md's Hooks section for the full call-site and
        keyword-argument contract). Two dispatch sites are not covered by
        the loop's own error handling: ``agent_start`` fires before the
        turn loop's error handling exists, so a handler exception there
        propagates straight out of this generator instead of becoming an
        :class:`ErrorMessage`; and ``turn_end`` fires from a ``finally``
        wrapping each turn, so a ``turn_end`` handler's own exception
        replaces an in-flight ``return`` or exception rather than following
        it, per ordinary Python ``finally`` semantics. Returned hook messages
        are yielded directly; turn-end messages are suppressed during generator
        closure so cleanup never yields while handling GeneratorExit.

        Args:
            message: the initial user message that starts the turn.
            prelude: messages appended to history right after ``message``,
                in order, before the turn loop's first request is built, so
                the model sees them as part of the same turn. Unlike
                ``message``, each is yielded. Defaults to none.

        Yields:
            Each ``prelude`` message, then each message produced during the
            run, including intermediate tool calls and tool results. The
            initial ``message`` is appended to history but never yielded, since
            the caller already holds it.
        """
        yield from self._hook_manager.run(Hook.AGENT_START, message=message)
        self.add_message_to_history(message)
        for extra in prelude or []:
            yield from self._emit(extra)

        # backoff wait that grows by the multiplier on each consecutive retry
        retry_wait = self._retry_delay
        consecutive_errors = 0

        # unlimited runs iterate until the model stops requesting tools
        turns = (
            count()
            if self._max_turns == AGENT_UNLIMITED_MAX_TURNS
            else range(self._max_turns)
        )
        for turn in turns:
            tool_calls: List[ToolCallMessage] = []
            received_finalized = False
            request = None

            # try sending request to the client, if something goes wrong, retry
            try:
                yield from self._hook_manager.run(Hook.TURN_START, turn=turn)

                system_prompt = self._system_prompt_builder.build()
                yield from self._hook_manager.run(
                    Hook.SYSTEM_PROMPT_BUILD, system_prompt=system_prompt
                )
                request = self._client.build_request(
                    messages=self._message_history,
                    system_prompt=system_prompt,
                    tools=self._tool_schemas,
                )
                yield from self._hook_manager.run(
                    Hook.BEFORE_MODEL_REQUEST, request=request
                )

                for response_message in self._client.send_request(request=request):
                    if isinstance(response_message, ChunkMessage):
                        yield response_message
                        continue
                    received_finalized = True
                    if isinstance(response_message, ToolCallMessage):
                        tool_calls.append(response_message)
                    yield from self._hook_manager.run(
                        Hook.MESSAGE_RECEIVED, response_message=response_message
                    )
                    yield from self._emit(response_message)

                yield from self._hook_manager.run(
                    Hook.AFTER_MODEL_RESPONSE, request=request, tool_calls=tool_calls
                )

                # a completed stream is a successful turn: reset the
                # consecutive-error tracking and the backoff wait
                consecutive_errors = 0
                retry_wait = self._retry_delay

                # natural stop: text-only reply
                if not tool_calls:
                    yield from self._hook_manager.run(
                        Hook.AGENT_END, message_history=self._message_history
                    )
                    return

                # call the tools if tools are requested
                for call in tool_calls:
                    decision, hook_messages = self._permission_manager.validate(call)
                    yield from hook_messages
                    feedback = self._permission_message(call, decision)
                    if decision.outcome is PermissionOutcome.ALLOWED:
                        result, hook_messages = self._run_tool(call)
                        yield from hook_messages
                        yield from self._emit(result)

                    # the note follows the result; a refusal is fed back to the
                    # model and surfaced to the user so the model can react
                    if feedback:
                        yield from self._emit(feedback)

            except Exception as e:
                # dispatched for every failed turn, retried or not; the
                # source="agent" dispatch below additionally fires only when
                # the run is about to end
                yield from self._hook_manager.run(
                    Hook.ERROR, source="model", error=e, request=request
                )
                # always surface the error to the consumer for visibility, but
                # only retry the turn when nothing finalized came through;
                # post-stream errors that fire after finalized messages were
                # already delivered (e.g. provider-side logging callback bugs)
                # must not trigger a retry
                yield ErrorMessage(content=f"Error while executing user request: {e}")
                if received_finalized:
                    yield from self._hook_manager.run(
                        Hook.ERROR, source="agent", error=e
                    )
                    return
                # abort once the failed turns in a row reach max_retries
                consecutive_errors += 1
                if consecutive_errors >= self._max_retries:
                    yield ErrorMessage(content="Max retries exceeded")
                    yield from self._hook_manager.run(
                        Hook.ERROR, source="agent", error=e
                    )
                    return
                # wait before retrying so the turn is not retried immediately;
                # the wait grows by the multiplier on each consecutive retry
                time.sleep(retry_wait)
                retry_wait *= self._retry_delay_multiplier

            finally:
                # closing a generator must run cleanup without yielding
                closing = isinstance(sys.exception(), GeneratorExit)
                hook_messages = self._hook_manager.run(Hook.TURN_END, turn=turn)
                if not closing:
                    yield from hook_messages

        yield from self._hook_manager.run(Hook.ERROR, source="agent", error=None)
        yield ErrorMessage(content="Max turns exceeded")
