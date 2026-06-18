"""Agent loop orchestrating client calls, tool use and message history."""

import time
from collections.abc import Iterator
from typing import Dict, List, Type

from arancio.core.builders.system_prompt import SystemPromptBuilder
from arancio.core.clients.base import BaseClient
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
        permission_manager: PermissionManager | None = None,
        max_turns: int = 1000,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize the agent with a client and permission manager.

        Args:
            client: the LLM client used to send requests.
            permission_manager: the permission manager that creates the agent's
                tools and gates each tool call. Defaults to a fully granted
                manager (every category at ``ASK``) when omitted.
            max_turns: the maximum number of loop turns before aborting.
            retry_delay: seconds to wait before retrying a failed turn, so the
                turn is not retried immediately.
        """
        self._client: BaseClient = client
        self._max_turns: int = max_turns
        self._retry_delay: float = retry_delay
        self._message_history: List[Message] = []
        self._system_prompt_builder: Type[SystemPromptBuilder] = SystemPromptBuilder

        if not permission_manager:
            permission_manager = PermissionManager()
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
            f"tools={list(self._tools)!r}"
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

    def add_tool(self, tool: BaseTool):
        """Append a tool to the agent's tool catalog.

        Args:
            tool: the tool definition to expose to the model.
        """
        self._tools[tool.name] = tool

    def _refresh_tools(self) -> None:
        """Rebuild the tool catalog from the permission manager's grants."""
        self._tools = {
            tool.name: tool for tool in self._permission_manager.allowed_tools()
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
                # wait before retrying so the turn is not retried immediately
                time.sleep(self._retry_delay)

        yield ErrorMessage(content="Max turns exceeded")
