"""Agent loop orchestrating client calls, tool use and message history."""

import sys
import time
from collections.abc import Generator, Iterator
from itertools import count
from typing import List

from arancio.core.agents.constants import (
    DEFAULT_TURN,
    UNLIMITED_MAX_TURNS,
)
from arancio.core.agents.types import Turn
from arancio.core.builders.system_prompt import SystemPromptBuilder
from arancio.core.clients.base import Client
from arancio.core.events import CoreAnswer, CoreEvent
from arancio.core.hooks.base import Hook
from arancio.core.hooks.manager import HookManager
from arancio.core.messages.base import ChunkMessage, ErrorMessage, Message
from arancio.core.messages.tool import (
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
)
from arancio.core.messages.user import UserMessage
from arancio.core.permissions.manager import PermissionManager
from arancio.core.permissions.types import (
    PermissionCategory,
    PermissionDecision,
    PermissionLevel,
    PermissionOutcome,
)
from arancio.core.tools.manager import ToolManager


class Agent:
    """Coding agent driving a multi-turn loop against an LLM client.

    Attributes:
        system_prompt: rendered text for the current turn, editable during
            SYSTEM_PROMPT_BUILD before the request is built. Initially empty.
        turn_index: current zero-based loop index. Set at each iteration; edits
            remain visible for that iteration without changing the loop's
            scheduling.
    """

    def __init__(
        self,
        client: Client,
        permission_manager: PermissionManager,
        tool_manager: ToolManager,
        hook_manager: HookManager,
        message_history: List[Message] | None = None,
        turn: Turn = DEFAULT_TURN,
    ) -> None:
        """Initialize the agent with its client and managers.

        Args:
            client: the LLM client used to send requests.
            permission_manager: the permission manager holding the grants and
                resolving each tool call.
            tool_manager: the tool catalog; the agent asks it for schemas and
                categories and runs every allowed call through it, refreshing
                its active tools from the allowed categories.
            hook_manager: the hook manager the loop and system-prompt build
                dispatch through (see AGENTS.md's Hooks section); the agent
                provides itself and ``client`` as ``agent`` and
                ``client``.
            message_history: the model context the agent starts from, copied so
                the caller keeps no handle on it. Defaults to none, an empty
                history; a restored session passes its saved messages here.
            turn: the turn limit (``"inf"`` by default, so the loop runs until
                the model stops requesting tools) and the retry rules for failed
                turns (exponential backoff).
        """
        self.system_prompt: str = ""
        self.turn_index: int = 0
        self._client: Client = client
        self._turn: Turn = turn
        self._message_history: List[Message] = list(message_history or [])
        self._system_prompt_builder: SystemPromptBuilder = SystemPromptBuilder()
        self._permission_manager: PermissionManager = permission_manager
        self._tool_manager: ToolManager = tool_manager
        self._hook_manager: HookManager = hook_manager
        self._hook_manager.provide(agent=self, client=client)

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
            f"turn={self._turn!r}, "
            f"tool_manager={self._tool_manager!r}"
            ")"
        )

    def __str__(self) -> str:
        """Return a pretty, multi-line representation of the agent.

        Returns:
            An indented, one-field-per-line rendering suitable for printing.
        """
        return (
            f"{type(self).__name__}(\n"
            f"    client={self._client!r},\n"
            f"    permissions={self._permission_manager!r},\n"
            f"    turn={self._turn!r},\n"
            f"    tool_manager={self._tool_manager!r},\n"
            ")"
        )

    @property
    def turn(self) -> Turn:
        """The turn limit and the retry rules for failed turns.

        Returns:
            The frozen configuration the loop reads.
        """
        return self._turn

    @turn.setter
    def turn(self, value: Turn) -> None:
        """Replace the turn limit and the retry rules.

        Args:
            value: the new frozen configuration.
        """
        self._turn = value

    def _refresh_tools(self) -> None:
        """Rebuild the active tools from the permission manager's grants."""
        self._tool_manager.refresh(self._permission_manager.allowed)

    def add(
        self,
        permission_category: PermissionCategory,
        permission_level: PermissionLevel = PermissionLevel.ASK,
    ) -> None:
        """Grant a category and rebuild the tool catalog.

        Args:
            permission_category: the category to grant.
            permission_level: the permission level. Defaults to
                :attr:`PermissionLevel.ASK`.
        """
        self._permission_manager.add(permission_category, permission_level)
        self._refresh_tools()

    def remove(self, permission_category: PermissionCategory) -> None:
        """Revoke a category's grant and rebuild the tool catalog.

        Args:
            permission_category: the category whose grant is removed.
        """
        self._permission_manager.remove(permission_category)
        self._refresh_tools()

    def set_level(
        self, permission_category: PermissionCategory, permission_level: PermissionLevel
    ) -> None:
        """Change a category's permission level and rebuild the tool catalog.

        Args:
            permission_category: the category whose grant to update.
            permission_level: the new permission level.
        """
        self._permission_manager.set_level(permission_category, permission_level)
        self._refresh_tools()

    def set(self, permissions: dict[PermissionCategory, PermissionLevel]) -> None:
        """Replace all permission grants and rebuild the tool catalog.

        Args:
            permissions: the new category-to-level grants, replacing the
                current ones wholesale.
        """
        self._permission_manager.set(permissions)
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
        tool_call: ToolCallMessage, permission_decision: PermissionDecision
    ) -> Message | None:
        """Build the model-facing message a resolved permission decision needs.

        Args:
            tool_call: the tool call the decision resolves.
            permission_decision: the permission decision to word for the model.

        Returns:
            The note wrapped in a report-then-answer instruction when the call
            is allowed with one, the tool error describing the denial or the
            missing tool when it is refused, and ``None`` for a plain allow.
        """
        if permission_decision.outcome is PermissionOutcome.ALLOWED:
            if not permission_decision.note:
                return None

            # the note instructs the model to report the result first, then
            # answer it, so it is sent after the result
            instruction = (
                f"While running tool {tool_call.id}: {tool_call.name}, "
                f"user also noted: {permission_decision.note}. "
                f"First report tool calling result, then answer user note."
            )
            return UserMessage(
                content=instruction, display_text=permission_decision.note
            )

        # a missing tool must not read as a user denial: the model asked for a
        # tool that is not part of this run at all
        if permission_decision.outcome is PermissionOutcome.UNAVAILABLE:
            return ToolErrorMessage(
                content=f"Tool {tool_call.name} does not exist.", id=tool_call.id
            )

        content = f"Tool call {tool_call.name} denied by user."
        if permission_decision.note:
            content += f" Additional information from user: {permission_decision.note}"
        return ToolErrorMessage(content=content, id=tool_call.id)

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
        self, tool_call: ToolCallMessage
    ) -> tuple[ToolResultMessage, list[Message]]:
        """Run one tool call through the tool manager.

        Args:
            tool_call: the tool call to run.

        Returns:
            The tool result and hook messages.
        """
        return self._tool_manager.run_tool_call(tool_call)

    def __call__(
        self,
        message: Message,
        prelude: List[Message] | None = None,
    ) -> Generator[CoreEvent, CoreAnswer, None]:
        """Run the agent loop starting from the given user message.

        The run is a generator driven by the caller: start it with ``next()``
        or ``.send(None)``, continue with ``.send(None)`` after a message, and
        answer a :class:`~arancio.core.events.PermissionRequest` with its
        matching :class:`~arancio.core.events.PermissionResponse`. The value
        returned by ``.send()`` is the next event. A wrong or missing answer
        never approves a call: it raises a protocol error, which ends the run
        with an :class:`ErrorMessage` and no model retry.

        The user message, every parsed client message, and every local
        tool result are appended to conversation history. Produced client
        and tool-result messages are yielded in order as they are handled.
        The loop ends naturally when the model replies without tool calls;
        when ``turn.max`` is an integer, it also aborts after that many
        turns, and it always aborts after ``turn.retry.max`` consecutive
        failed turns.

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
        closure so cleanup never yields.

        Args:
            message: the initial user message that starts the turn.
            prelude: messages appended to history right after ``message``,
                in order, before the turn loop's first request is built, so
                the model sees them as part of the same turn. Unlike
                ``message``, each is yielded. Defaults to none.

        Yields:
            Each ``prelude`` message, then each message produced during the
            run, including intermediate tool calls and tool results, and a
            permission request for each tool call that needs an answer. The
            initial ``message`` is appended to history but never yielded,
            since the caller already holds it.
        """
        yield from self._hook_manager.run(Hook.AGENT_START, message=message)
        self.add_message_to_history(message)
        for extra in prelude or []:
            yield from self._emit(extra)

        # backoff wait that grows by the multiplier on each consecutive retry
        retry_wait = self._turn.retry.delay
        consecutive_errors = 0

        # unlimited runs iterate until the model stops requesting tools
        turns = (
            count() if self._turn.max == UNLIMITED_MAX_TURNS else range(self._turn.max)
        )
        for turn_index in turns:
            self.turn_index = turn_index
            tool_calls: List[ToolCallMessage] = []
            received_finalized = False
            request = None

            # try sending request to the client, if something goes wrong, retry
            try:
                yield from self._hook_manager.run(Hook.TURN_START)

                self.system_prompt = self._system_prompt_builder.build()
                yield from self._hook_manager.run(Hook.SYSTEM_PROMPT_BUILD)
                request = self._client.build_request(
                    messages=self._message_history,
                    system_prompt=self.system_prompt,
                    tools=self._tool_manager.tool_schemas,
                )
                yield from self._hook_manager.run(
                    Hook.BEFORE_MODEL_REQUEST, request=request
                )

                # close the client stream when the turn ends or the run is closed
                stream = self._client.send_request(request=request)
                try:
                    for response_message in stream:
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
                finally:
                    stream.close()

                yield from self._hook_manager.run(
                    Hook.AFTER_MODEL_RESPONSE, request=request, tool_calls=tool_calls
                )

                # a completed stream is a successful turn: reset the
                # consecutive-error tracking and the backoff wait
                consecutive_errors = 0
                retry_wait = self._turn.retry.delay

                # natural stop: text-only reply
                if not tool_calls:
                    yield from self._hook_manager.run(
                        Hook.AGENT_END, message_history=self._message_history
                    )
                    return

                # call the tools if tools are requested
                for tool_call in tool_calls:
                    permission_category = self._tool_manager.get_permission_category(
                        tool_call.name
                    )
                    permission_decision, hook_messages = yield from (
                        self._permission_manager.validate(
                            tool_call, permission_category
                        )
                    )
                    yield from hook_messages
                    feedback = self._permission_message(tool_call, permission_decision)
                    if permission_decision.outcome is PermissionOutcome.ALLOWED:
                        result, hook_messages = self._run_tool(tool_call)
                        yield from hook_messages
                        yield from self._emit(result)

                    # the note follows the result; a refusal is fed back to the model
                    # and surfaced to the user so the model can react
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
                # must not trigger a retry, and neither must a failure in the
                # tool phase, such as a wrong permission answer
                yield ErrorMessage(content=f"Error while executing user request: {e}")
                if received_finalized:
                    yield from self._hook_manager.run(
                        Hook.ERROR, source="agent", error=e
                    )
                    return
                # abort once the failed turns in a row reach the retry maximum
                consecutive_errors += 1
                if consecutive_errors >= self._turn.retry.max:
                    yield ErrorMessage(content="Max retries exceeded")
                    yield from self._hook_manager.run(
                        Hook.ERROR, source="agent", error=e
                    )
                    return
                # wait before retrying so the turn is not retried immediately;
                # the wait grows by the multiplier on each consecutive retry
                time.sleep(retry_wait)
                retry_wait *= self._turn.retry.multiplier

            finally:
                # closing a run must clean up without yielding
                closing = isinstance(sys.exception(), GeneratorExit)
                hook_messages = self._hook_manager.run(Hook.TURN_END)
                if not closing:
                    yield from hook_messages

        yield from self._hook_manager.run(Hook.ERROR, source="agent", error=None)
        yield ErrorMessage(content="Max turns exceeded")
