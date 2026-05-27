"""Agent loop orchestrating client calls, tool use and message history."""

from collections.abc import Iterator
from typing import Dict, List, Type

from codo.builders.system_prompt import SystemPromptBuilder
from codo.clients.base import BaseClient
from codo.exceptions import MaxTurnsException
from codo.tools.base import BaseTool
from codo.types.messages import (
    ChunkMessage,
    ErrorMessage,
    Message,
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
)
from codo.types.tools import ToolSchema


class Agent:
    """Coding agent driving a multi-turn loop against an LLM client."""

    def __init__(
        self,
        client: BaseClient,
        tools: List[BaseTool] | None = None,
        max_turns: int = 1000,
    ) -> None:
        """Initialize the agent with a client, model and tool catalog.

        Args:
            client: the LLM client used to send requests.
            tools: the tool catalog available to the model; when None
                the agent runs without tools.
            max_turns: the maximum number of loop turns before aborting.
        """
        self._client: BaseClient = client
        self._max_turns: int = max_turns
        self._message_history: List[Message] = []
        self._system_prompt_builder: Type[SystemPromptBuilder] = SystemPromptBuilder

        if not tools:
            tools = []
        self._tools: Dict[str, BaseTool] = {tool.name: tool for tool in tools}

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the agent.

        Returns:
            A compact string with configuration and state counts.
        """
        return (
            f"{type(self).__name__}("
            f"client={self._client!r}, "
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
                output=message,
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

        Raises:
            MaxTurnsException: when the turn limit is exhausted.
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

            except Exception as e:
                # always surface the error to the consumer for visibility,
                # but only retry the turn when nothing finalized came through;
                # post-stream errors that fire after finalized messages were
                # already delivered (e.g. provider-side logging callback bugs)
                # must not trigger a retry
                yield ErrorMessage(content=f"Error while executing user request: {e}")
                if not received_finalized:
                    continue

            # no response from the client, something happened so retry
            if not received_finalized:
                continue

            # natural stop: text-only reply
            if not tool_calls:
                return

            # call the tools if tools are requested
            for call in tool_calls:
                tool_result = self._run_tool(call)
                self._add_message_to_history(tool_result)
                yield tool_result

        message = "Max turns exceeded"
        yield ErrorMessage(content=message)
        raise MaxTurnsException(message)
