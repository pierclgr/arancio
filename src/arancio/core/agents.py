"""Agent loop orchestrating client calls, tool use and message history."""

import time
from collections.abc import Iterator
from typing import Dict, List, Type

from arancio.core.builders.system_prompt import SystemPromptBuilder
from arancio.core.clients.base import BaseClient
from arancio.core.constants.agent import (
    AGENT_DEFAULT_MAX_TURNS,
    AGENT_DEFAULT_TURN_WAIT_TIME,
    AGENT_DEFAULT_TURN_WAIT_TIME_MULTIPLIER,
)
from arancio.core.permissions.manager import PermissionManager
from arancio.core.tools.base import BaseTool
from arancio.core.types.messages import (
    ChunkMessage,
    ErrorMessage,
    Message,
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
)
from arancio.core.types.permissions import PermissionCategory, PermissionLevel
from arancio.core.types.tools import ToolSchema


class Agent:
    """Coding agent driving a multi-turn loop against an LLM client."""

    def __init__(
        self,
        client: BaseClient,
        permission_manager: PermissionManager,
        max_turns: int = AGENT_DEFAULT_MAX_TURNS,
        retry_delay: float = AGENT_DEFAULT_TURN_WAIT_TIME,
        retry_delay_multiplier: float = AGENT_DEFAULT_TURN_WAIT_TIME_MULTIPLIER,
    ) -> None:
        """Initialize the agent with a client and permission manager.

        Args:
            client: the LLM client used to send requests.
            permission_manager: the permission manager that creates the agent's
                tools and gates each tool call.
            max_turns: the maximum number of loop turns before aborting.
            retry_delay: seconds to wait before retrying a failed turn, so the
                turn is not retried immediately.
            retry_delay_multiplier: factor the retry wait grows by on each
                consecutive failed-turn retry (exponential backoff).
        """
        self._client: BaseClient = client
        self._max_turns: int = max_turns
        self._retry_delay: float = retry_delay
        self._retry_delay_multiplier: float = retry_delay_multiplier
        self._message_history: List[Message] = []
        self._system_prompt_builder: Type[SystemPromptBuilder] = SystemPromptBuilder
        self._permission_manager: PermissionManager = permission_manager

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
            f"    tools={tools_block},\n"
            ")"
        )

    @property
    def _tool_schemas(self) -> List[ToolSchema]:
        return [tool.schema() for tool in self._tools.values()]

    @property
    def _system_prompt(self) -> str:
        """Build and return the current system prompt.

        Returns:
            The rendered system prompt string.
        """
        return self._system_prompt_builder.build()

    @property
    def max_turns(self) -> int:
        """Return the maximum number of loop turns per run.

        Returns:
            The maximum number of turns before the run aborts.
        """
        return self._max_turns

    @max_turns.setter
    def max_turns(self, value: int) -> None:
        """Set the maximum number of loop turns per run.

        Args:
            value: the new maximum number of turns.
        """
        self._max_turns = value

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
            tool.name: tool for tool in self._permission_manager.get_allowed_tools
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

    def _add_message_to_history(self, message: Message) -> None:
        """Append a single message to the conversation history.

        Args:
            message: the message to add.
        """
        self._message_history.append(message)

    def _run_tool(self, call: ToolCallMessage) -> ToolResultMessage:
        """Execute a tool call and return a ToolResultMessage.

        Args:
            call: the tool call message to execute.

        Returns:
            A ToolResultMessage with the tool output and error status.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            message = f"Unknown tool {call.name}"
            return ToolErrorMessage(
                content=message,
                id=call.id,
            )
        else:
            return tool.call(call_id=call.id, **(call.arguments or {}))

    def run(self, message: Message) -> Iterator[Message]:
        """Run the agent loop starting from the given user message.

        The user message, every parsed client message, and every local
        tool result are appended to conversation history. Produced client
        and tool-result messages are yielded in order as they are handled.

        Args:
            message: the initial user message that starts the turn.

        Yields:
            Each message produced during the run, including intermediate
            tool calls and tool results.
        """
        self._add_message_to_history(message=message)

        # backoff wait that grows by the multiplier on each consecutive retry
        retry_wait = self._retry_delay

        for _ in range(self._max_turns):
            tool_calls: List[ToolCallMessage] = []
            received_finalized = False

            # try sending request to the client, if something goes wrong, retry
            try:
                request = self._client.build_request(
                    messages=self._message_history,
                    system_prompt=self._system_prompt,
                    tools=self._tool_schemas,
                )

                for response_message in self._client.send_request(request=request):
                    if not isinstance(response_message, ChunkMessage):
                        received_finalized = True
                        self._add_message_to_history(message=response_message)
                        if isinstance(response_message, ToolCallMessage):
                            tool_calls.append(response_message)
                    yield response_message

                # natural stop: text-only reply
                if not tool_calls:
                    return

                # call the tools if tools are requested
                for call in tool_calls:
                    authorized, feedback = self._permission_manager.validate(call)
                    if authorized:
                        tool_result = self._run_tool(call)
                        self._add_message_to_history(tool_result)
                        yield tool_result

                        # a note carries an instruction telling the model to
                        # report the result first, then answer it; send it as a
                        # separate message after the result
                        if feedback:
                            self._add_message_to_history(feedback)
                            yield feedback
                    else:
                        # denied or not permitted: feed the message back to the
                        # model and surface it to the user so the model can react
                        self._add_message_to_history(feedback)
                        yield feedback

            except Exception as e:
                # always surface the error to the consumer for visibility, but
                # only retry the turn when nothing finalized came through;
                # post-stream errors that fire after finalized messages were
                # already delivered (e.g. provider-side logging callback bugs)
                # must not trigger a retry
                yield ErrorMessage(content=f"Error while executing user request: {e}")
                if received_finalized:
                    return
                # wait before retrying so the turn is not retried immediately;
                # the wait grows by the multiplier on each consecutive retry
                time.sleep(retry_wait)
                retry_wait *= self._retry_delay_multiplier

        yield ErrorMessage(content="Max turns exceeded")
