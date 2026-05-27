"""Tests for agent loop behavior."""

from typing import List

import pytest

from codo.agents import Agent
from codo.clients.base import BaseClient
from codo.exceptions import MaxTurnsException
from codo.parsers.tool_result.base import BaseToolResultParser
from codo.tools.base import BaseTool
from codo.types.messages import (
    AssistantChunkMessage,
    AssistantMessage,
    ErrorMessage,
    Message,
    ReasoningMessage,
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
    UserMessage,
)
from codo.types.requests import BaseRequest


class _ReasoningClient(BaseClient):
    """Client returning reasoning state plus a visible assistant message."""

    _model_options = ["fake-model"]
    _thinking_options: List[str] = []
    _default_model_id = "fake-model"
    _default_thinking_effort = None
    request_schema = BaseRequest

    def __init__(self) -> None:
        """Initialize the fake client."""
        super().__init__(token="token")
        self.requests: List[BaseRequest] = []

    def send_request(self, request: BaseRequest) -> List[Message]:
        """Record the request and return a reasoning item plus text.

        Args:
            request: the request built by the agent.

        Returns:
            A reasoning message followed by a visible assistant message.
        """
        self.requests.append(request)
        return [
            ReasoningMessage(
                item={
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "encrypted",
                },
                content="",
            ),
            AssistantMessage(content="visible"),
        ]


class _ToolLoopClient(_ReasoningClient):
    """Client returning a tool call before the final assistant message."""

    def send_request(self, request: BaseRequest) -> List[Message]:
        """Return a tool call on first request, then a final message.

        Args:
            request: the request built by the agent.

        Returns:
            A tool-call batch followed by a final assistant batch.
        """
        self.requests.append(request)
        if len(self.requests) == 1:
            return [
                ToolCallMessage(
                    content='EchoTool({"text": "hello"})',
                    id="call_1",
                    name="EchoTool",
                    arguments={"text": "hello"},
                )
            ]
        return [AssistantMessage(content="done")]


class _StreamingClient(_ReasoningClient):
    """Client returning streamed text before the finalized assistant message."""

    def send_request(self, request: BaseRequest) -> List[Message]:
        """Return a text chunk followed by the finalized message.

        Args:
            request: the request built by the agent.

        Returns:
            A chunk message followed by a final assistant message.
        """
        self.requests.append(request)
        return [
            AssistantChunkMessage(content="vis"),
            AssistantMessage(content="visible"),
        ]


class _EchoToolResultParser(BaseToolResultParser):
    """Parse echo output into a tool result message."""

    @classmethod
    def parse(
        cls,
        call_id: str,
        output: str,
        is_error: bool = False,
    ) -> ToolResultMessage:
        """Parse echo output.

        Args:
            call_id: identifier of the tool call this result answers.
            output: raw echo output.
            is_error: whether the tool execution failed.

        Returns:
            A parsed tool result message.
        """
        result_class = ToolErrorMessage if is_error else ToolResultMessage
        return result_class(
            content=output,
            id=call_id,
            output=output,
        )


class EchoTool(BaseTool):
    """Tool returning the text passed to it."""

    _result_parser = _EchoToolResultParser

    def __init__(self) -> None:
        """Set description and input schema directly, skipping disk load."""
        self.description = "Echo text."
        self.input_schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }

    def _call(self, **kwargs) -> str:
        """Return the provided text.

        Args:
            **kwargs: tool arguments.

        Returns:
            The input text.
        """
        return kwargs["text"]


class FailingTool(EchoTool):
    """Tool raising during execution."""

    def _call(self, **kwargs) -> str:
        """Raise for every invocation.

        Args:
            **kwargs: tool arguments.

        Raises:
            RuntimeError: always raised to simulate tool failure.
        """
        raise RuntimeError("boom")


class _UnknownToolClient(_ReasoningClient):
    """Client requesting a tool that is not registered."""

    def send_request(self, request: BaseRequest) -> List[Message]:
        """Return an unknown tool call, then a final message.

        Args:
            request: the request built by the agent.

        Returns:
            A tool-call batch followed by a final assistant batch.
        """
        self.requests.append(request)
        if len(self.requests) == 1:
            return [
                ToolCallMessage(
                    content='MissingTool({"text": "hello"})',
                    id="call_1",
                    name="MissingTool",
                    arguments={"text": "hello"},
                )
            ]
        return [AssistantMessage(content="done")]


class _FailingToolClient(_ReasoningClient):
    """Client requesting a tool that raises."""

    def send_request(self, request: BaseRequest) -> List[Message]:
        """Return a failing tool call, then a final message.

        Args:
            request: the request built by the agent.

        Returns:
            A tool-call batch followed by a final assistant batch.
        """
        self.requests.append(request)
        if len(self.requests) == 1:
            return [
                ToolCallMessage(
                    content='FailingTool({"text": "hello"})',
                    id="call_1",
                    name="FailingTool",
                    arguments={"text": "hello"},
                )
            ]
        return [AssistantMessage(content="done")]


class _FailingClient(_ReasoningClient):
    """Client raising once before returning a final assistant message."""

    def send_request(self, request: BaseRequest) -> List[Message]:
        """Raise on the first request, then return a final message.

        Args:
            request: the request built by the agent.

        Returns:
            A final assistant message after the first failure.

        Raises:
            RuntimeError: on the first request.
        """
        self.requests.append(request)
        if len(self.requests) == 1:
            raise RuntimeError("network down")
        return [AssistantMessage(content="done")]


class _PostFinalizeFailingClient(_ReasoningClient):
    """Client yielding a finalized message then raising during iteration.

    Simulates upstream libraries (e.g. LiteLLM logging callbacks) that crash *after* the
    stream has already delivered all finalized output, which must not cause the agent to
    retry the turn.
    """

    def send_request(self, request: BaseRequest):
        """Yield the finalized assistant message, then raise on next iteration.

        Args:
            request: the request built by the agent.

        Yields:
            A finalized assistant message before raising.

        Raises:
            RuntimeError: after the finalized message has been yielded.
        """
        self.requests.append(request)
        yield AssistantMessage(content="visible")
        raise RuntimeError("post-stream logging blew up")


def test_agent_stores_and_returns_reasoning_messages() -> None:
    """Reasoning messages stay in history and are yielded to the caller."""
    client = _ReasoningClient()
    agent = Agent(client=client)

    response = list(agent.run(UserMessage(content="hello")))

    assert response == [
        ReasoningMessage(
            item={
                "id": "rs_1",
                "type": "reasoning",
                "summary": [],
                "encrypted_content": "encrypted",
            },
            content="",
        ),
        AssistantMessage(content="visible"),
    ]
    assert any(isinstance(msg, ReasoningMessage) for msg in agent._message_history)


def test_agent_yields_chunks_without_storing_them() -> None:
    """Streaming chunks are yielded but omitted from provider history."""
    client = _StreamingClient()
    agent = Agent(client=client)

    response = list(agent.run(UserMessage(content="hello")))

    assert response == [
        AssistantChunkMessage(content="vis"),
        AssistantMessage(content="visible"),
    ]
    assert AssistantChunkMessage(content="vis") not in agent._message_history
    assert AssistantMessage(content="visible") in agent._message_history


def test_agent_returns_tool_call_result_and_final_response() -> None:
    """Agent run yields all messages produced during the turn."""
    client = _ToolLoopClient()
    agent = Agent(client=client, tools=[EchoTool()])

    response = list(agent.run(UserMessage(content="hello")))

    assert response == [
        ToolCallMessage(
            content='EchoTool({"text": "hello"})',
            id="call_1",
            name="EchoTool",
            arguments={"text": "hello"},
        ),
        ToolResultMessage(content="hello", id="call_1", output="hello"),
        AssistantMessage(content="done"),
    ]


def test_agent_yields_error_message_for_unknown_tool() -> None:
    """Unknown tool calls yield tool errors to the consumer."""
    client = _UnknownToolClient()
    agent = Agent(client=client)

    response = list(agent.run(UserMessage(content="hello")))

    assert response == [
        ToolCallMessage(
            content='MissingTool({"text": "hello"})',
            id="call_1",
            name="MissingTool",
            arguments={"text": "hello"},
        ),
        ToolErrorMessage(
            content="Unknown tool MissingTool",
            id="call_1",
            output="Unknown tool MissingTool",
        ),
        AssistantMessage(content="done"),
    ]
    assert (
        ToolErrorMessage(
            content="Unknown tool MissingTool",
            id="call_1",
            output="Unknown tool MissingTool",
        )
        in agent._message_history
    )


def test_agent_yields_error_message_for_failing_tool() -> None:
    """Tool exceptions yield tool errors to the consumer."""
    client = _FailingToolClient()
    agent = Agent(client=client, tools=[FailingTool()])

    response = list(agent.run(UserMessage(content="hello")))

    assert response == [
        ToolCallMessage(
            content='FailingTool({"text": "hello"})',
            id="call_1",
            name="FailingTool",
            arguments={"text": "hello"},
        ),
        ToolErrorMessage(
            content="Error while executing FailingTool: boom",
            id="call_1",
            output="Error while executing FailingTool: boom",
        ),
        AssistantMessage(content="done"),
    ]
    assert (
        ToolErrorMessage(
            content="Error while executing FailingTool: boom",
            id="call_1",
            output="Error while executing FailingTool: boom",
        )
        in agent._message_history
    )


def test_agent_yields_error_message_for_loop_exception() -> None:
    """Agent-loop exceptions are yielded as error messages."""
    client = _FailingClient()
    agent = Agent(client=client)

    response = list(agent.run(UserMessage(content="hello")))

    assert response == [
        ErrorMessage(content="Error while executing user request: network down"),
        AssistantMessage(content="done"),
    ]
    assert all(not isinstance(msg, ErrorMessage) for msg in agent._message_history)


def test_agent_swallows_post_finalize_exception_without_retry() -> None:
    """Exceptions raised after a finalized message ends the turn cleanly."""
    client = _PostFinalizeFailingClient()
    agent = Agent(client=client)

    response = list(agent.run(UserMessage(content="hello")))

    assert response == [AssistantMessage(content="visible")]
    assert len(client.requests) == 1
    assert AssistantMessage(content="visible") in agent._message_history
    assert all(not isinstance(msg, ErrorMessage) for msg in agent._message_history)


def test_agent_yields_error_message_before_max_turns_exception() -> None:
    """Max turn exhaustion yields an error message before raising."""
    agent = Agent(client=_ToolLoopClient(), tools=[EchoTool()], max_turns=1)
    stream = agent.run(UserMessage(content="hello"))

    assert next(stream) == ToolCallMessage(
        content='EchoTool({"text": "hello"})',
        id="call_1",
        name="EchoTool",
        arguments={"text": "hello"},
    )
    assert next(stream) == ToolResultMessage(
        content="hello",
        id="call_1",
        output="hello",
    )
    assert next(stream) == ErrorMessage(content="Max turns exceeded")

    with pytest.raises(MaxTurnsException):
        next(stream)


def test_agent_yields_tool_result_before_next_model_request() -> None:
    """Tool results are yielded before the follow-up model request."""
    client = _ToolLoopClient()
    agent = Agent(client=client, tools=[EchoTool()])

    stream = agent.run(UserMessage(content="hello"))

    assert next(stream) == ToolCallMessage(
        content='EchoTool({"text": "hello"})',
        id="call_1",
        name="EchoTool",
        arguments={"text": "hello"},
    )
    assert len(client.requests) == 1

    assert next(stream) == ToolResultMessage(
        content="hello",
        id="call_1",
        output="hello",
    )
    assert len(client.requests) == 1

    assert next(stream) == AssistantMessage(content="done")
    assert len(client.requests) == 2
    assert list(stream) == []


def test_agent_repr_includes_configuration_without_history_contents() -> None:
    """Agent repr exposes debug state without dumping conversation content."""
    agent = Agent(client=_ReasoningClient(), max_turns=3)
    agent._add_message_to_history(UserMessage(content="secret"))

    assert repr(agent) == (
        "Agent("
        "client=_ReasoningClient("
        "model_id='fake-model', thinking_effort=None, thinking_summary=None"
        "), "
        "max_turns=3, "
        "tools=[]"
        ")"
    )
