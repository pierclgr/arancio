"""Tests for agent loop behavior."""

from typing import List

import pytest

from arancio.core.agents import Agent
from arancio.core.clients.base import BaseClient
from arancio.core.clients.litellm import LiteLLMClient
from arancio.core.controllers.requests import BaseControllerRequest
from arancio.core.controllers.responses import (
    BaseControllerResponse,
    Decision,
    PermissionResponse,
)
from arancio.core.parsers.tool_result.base import BaseToolResultParser
from arancio.core.permissions.manager import PermissionManager
from arancio.core.tools.base import BaseTool
from arancio.core.tools.manager import ToolManager
from arancio.core.types.messages import (
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
from arancio.core.types.permissions import PermissionCategory, PermissionLevel
from arancio.core.types.requests import BaseRequest


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


class _AlwaysFailingClient(_ReasoningClient):
    """Client raising on every request, so every turn retries."""

    def send_request(self, request: BaseRequest) -> List[Message]:
        """Raise on every request to force a retry on each turn.

        Args:
            request: the request built by the agent.

        Raises:
            RuntimeError: on every request.
        """
        self.requests.append(request)
        raise RuntimeError("always down")


class _IntermittentFailingClient(_ReasoningClient):
    """Client alternating one failed request with one successful turn."""

    def send_request(self, request: BaseRequest) -> List[Message]:
        """Raise on odd requests, answer with a tool call then a final reply.

        Args:
            request: the request built by the agent.

        Returns:
            A tool call on the second request, a final assistant message on
            the fourth.

        Raises:
            RuntimeError: on every odd-numbered request.
        """
        self.requests.append(request)
        turn = len(self.requests)
        if turn % 2 == 1:
            raise RuntimeError("intermittent")
        if turn == 2:
            return [
                ToolCallMessage(
                    content='EchoTool({"text": "hello"})',
                    id="call_1",
                    name="EchoTool",
                    arguments={"text": "hello"},
                )
            ]
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


class _LongToolLoopClient(_ReasoningClient):
    """Client requesting a tool call on every turn until the given count."""

    def __init__(self, tool_turns: int) -> None:
        """Initialize the client with the number of tool-calling turns.

        Args:
            tool_turns: how many consecutive requests answer with a tool call
                before the final text-only reply.
        """
        super().__init__()
        self._tool_turns = tool_turns

    def send_request(self, request: BaseRequest) -> List[Message]:
        """Return one tool call per request until exhausted, then finish.

        Args:
            request: the request built by the agent.

        Returns:
            A tool-call batch for each of the first ``tool_turns`` requests,
            then a final assistant batch.
        """
        self.requests.append(request)
        turn = len(self.requests)
        if turn <= self._tool_turns:
            return [
                ToolCallMessage(
                    content=f'EchoTool({{"text": "step {turn}"}})',
                    id=f"call_{turn}",
                    name="EchoTool",
                    arguments={"text": f"step {turn}"},
                )
            ]
        return [AssistantMessage(content="done")]


class _MultiToolLoopClient(_ReasoningClient):
    """Client returning two tool calls in one batch before the final message."""

    def send_request(self, request: BaseRequest) -> List[Message]:
        """Return two tool calls on the first request, then a final message.

        Args:
            request: the request built by the agent.

        Returns:
            A two-call batch followed by a final assistant batch.
        """
        self.requests.append(request)
        if len(self.requests) == 1:
            return [
                ToolCallMessage(
                    content='EchoTool({"text": "A"})',
                    id="a",
                    name="EchoTool",
                    arguments={"text": "A"},
                ),
                ToolCallMessage(
                    content='EchoTool({"text": "B"})',
                    id="b",
                    name="EchoTool",
                    arguments={"text": "B"},
                ),
            ]
        return [AssistantMessage(content="done")]


class _StubManager:
    """Minimal permission-manager stub exposing only the agent's two hooks.

    Decouples agent-loop tests from the real ToolManager: ``get_allowed_tools``
    returns a configured tool list and ``validate`` looks decisions up by call
    id then by tool name, defaulting to ``default``.
    """

    def __init__(
        self,
        tools: List[BaseTool] | None = None,
        decisions: dict[str, bool] | None = None,
        default: bool = True,
        message: Message | None = None,
    ) -> None:
        """Initialize the stub with the tools to expose and decision maps.

        Args:
            tools: the tools returned by ``get_allowed_tools``.
            decisions: per-call-id or per-name allow/deny answers for
                ``validate``.
            default: decision returned by ``validate`` when no entry matches.
            message: message returned by ``validate`` alongside its decision; a
                denial with no message falls back to a ``ToolErrorMessage``.
        """
        self._tools = tools or []
        self._decisions = decisions or {}
        self._default = default
        self._message = message

    @property
    def get_allowed_tools(self) -> List[BaseTool]:
        """Return the configured tools the agent may access.

        Returns:
            The tools this stub was configured with.
        """
        return self._tools

    def validate(self, call: ToolCallMessage) -> tuple[bool, Message | None]:
        """Return the configured decision and message for a call.

        Args:
            call: the tool call to decide on.

        Returns:
            An ``(allowed, message)`` pair; the decision is keyed by call id,
            then by tool name, else the default. A denial returns the
            configured message or a default ``ToolErrorMessage``.
        """
        if call.id in self._decisions:
            authorized = self._decisions[call.id]
        else:
            authorized = self._decisions.get(call.name, self._default)
        if authorized:
            return True, self._message
        if self._message is not None:
            return False, self._message
        denial = f"Tool call {call.name} denied by user."
        return False, ToolErrorMessage(content=denial, id=call.id)

    def __repr__(self) -> str:
        """Return a fixed representation for deterministic repr assertions.

        Returns:
            A constant string independent of instance state.
        """
        return "_StubManager()"


class _FakeController:
    """Controller stub approving every request (never asked in these tests)."""

    def request(self, request: BaseControllerRequest) -> BaseControllerResponse:
        """Approve any request.

        Args:
            request: the request to approve.

        Returns:
            An ALLOW permission response.
        """
        return PermissionResponse(decision=Decision.ALLOW)


def _summary_client() -> LiteLLMClient:
    """Build a summarization client for injected tools in tests.

    Returns:
        A non-streaming :class:`LiteLLMClient`.
    """
    return LiteLLMClient(model_id="ollama_chat/deepseek-v4-flash:cloud", stream=False)


def _agent(**kwargs) -> Agent:
    """Build an agent, defaulting ``permission_manager`` for tests.

    Args:
        **kwargs: keyword arguments forwarded to :class:`Agent`.

    Returns:
        An ``Agent`` with a default fully-granted ``permission_manager`` when
        none is given.
    """
    kwargs.setdefault(
        "permission_manager",
        PermissionManager(
            ToolManager(web_summary_client=_summary_client()), _FakeController()
        ),
    )
    return Agent(**kwargs)


def _permission_manager(
    permissions: dict[PermissionCategory, PermissionLevel] | None = None,
) -> PermissionManager:
    """Build a real permission manager backed by a tool manager for tests.

    Args:
        permissions: optional category-to-level grants forwarded to the
            manager.

    Returns:
        A :class:`PermissionManager` whose tool manager carries a summary
        client.
    """
    return PermissionManager(
        ToolManager(web_summary_client=_summary_client()),
        _FakeController(),
        permissions,
    )


def test_agent_stores_and_returns_reasoning_messages() -> None:
    """Reasoning messages stay in history and are yielded to the caller."""
    client = _ReasoningClient()
    agent = _agent(client=client, permission_manager=_StubManager())

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
    agent = _agent(client=client, permission_manager=_StubManager())

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
    agent = _agent(client=client, permission_manager=_StubManager(tools=[EchoTool()]))

    response = list(agent.run(UserMessage(content="hello")))

    assert response == [
        ToolCallMessage(
            content='EchoTool({"text": "hello"})',
            id="call_1",
            name="EchoTool",
            arguments={"text": "hello"},
        ),
        ToolResultMessage(content="hello", id="call_1"),
        AssistantMessage(content="done"),
    ]


def test_agent_yields_error_message_for_unknown_tool() -> None:
    """Unknown tool calls yield tool errors to the consumer."""
    client = _UnknownToolClient()
    agent = _agent(client=client, permission_manager=_StubManager())

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
        ),
        AssistantMessage(content="done"),
    ]
    assert (
        ToolErrorMessage(
            content="Unknown tool MissingTool",
            id="call_1",
        )
        in agent._message_history
    )


def test_agent_yields_error_message_for_failing_tool() -> None:
    """Tool exceptions yield tool errors to the consumer."""
    client = _FailingToolClient()
    agent = _agent(
        client=client, permission_manager=_StubManager(tools=[FailingTool()])
    )

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
        ),
        AssistantMessage(content="done"),
    ]
    assert (
        ToolErrorMessage(
            content="Error while executing FailingTool: boom",
            id="call_1",
        )
        in agent._message_history
    )


def test_agent_yields_error_message_for_loop_exception() -> None:
    """Agent-loop exceptions are yielded as error messages."""
    client = _FailingClient()
    agent = _agent(client=client, permission_manager=_StubManager(), retry_delay=0)

    response = list(agent.run(UserMessage(content="hello")))

    assert response == [
        ErrorMessage(content="Error while executing user request: network down"),
        AssistantMessage(content="done"),
    ]
    assert all(not isinstance(msg, ErrorMessage) for msg in agent._message_history)


def test_agent_retry_wait_grows_by_multiplier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each failed-turn retry waits the previous wait times the multiplier."""
    waits: list[float] = []
    monkeypatch.setattr("time.sleep", waits.append)

    agent = _agent(
        client=_AlwaysFailingClient(),
        permission_manager=_StubManager(),
        max_retries=4,
        retry_delay=1.0,
        retry_delay_multiplier=2.0,
    )

    list(agent.run(UserMessage(content="hello")))

    assert waits == [1.0, 2.0, 4.0]


def test_agent_yields_error_but_does_not_retry_after_finalized() -> None:
    """Exceptions after a finalized message surface as errors but end the turn."""
    client = _PostFinalizeFailingClient()
    agent = _agent(client=client, permission_manager=_StubManager())

    response = list(agent.run(UserMessage(content="hello")))

    assert response == [
        AssistantMessage(content="visible"),
        ErrorMessage(
            content="Error while executing user request: post-stream logging blew up"
        ),
    ]
    assert len(client.requests) == 1
    assert AssistantMessage(content="visible") in agent._message_history
    assert all(not isinstance(msg, ErrorMessage) for msg in agent._message_history)


def test_agent_unlimited_by_default_completes_long_tool_runs() -> None:
    """With the default (no limit) the loop runs until the model stops.

    Regression for the default budget being a small number: a legitimate
    multi-step task must never abort with "Max turns exceeded" unless the
    caller explicitly sets a limit.
    """
    client = _LongToolLoopClient(tool_turns=50)
    agent = _agent(client=client, permission_manager=_StubManager(tools=[EchoTool()]))
    assert agent.max_turns is None

    response = list(agent.run(UserMessage(content="hello")))

    assert response[-1] == AssistantMessage(content="done")
    assert all(not isinstance(msg, ErrorMessage) for msg in response)
    assert len(client.requests) == 51


def test_agent_yields_error_message_when_max_retries_exceeded() -> None:
    """Reaching max_retries consecutive failed turns aborts the run."""
    client = _AlwaysFailingClient()
    agent = _agent(
        client=client,
        max_retries=2,
        permission_manager=_StubManager(),
        retry_delay=0,
    )

    response = list(agent.run(UserMessage(content="hello")))

    assert response == [
        ErrorMessage(content="Error while executing user request: always down"),
        ErrorMessage(content="Error while executing user request: always down"),
        ErrorMessage(content="Max retries exceeded"),
    ]
    assert len(client.requests) == 2


def test_agent_successful_turn_resets_consecutive_error_count() -> None:
    """A successful turn between failures resets the max_retries error budget."""
    client = _IntermittentFailingClient()
    agent = _agent(
        client=client,
        max_retries=2,
        permission_manager=_StubManager(tools=[EchoTool()]),
        retry_delay=0,
    )

    response = list(agent.run(UserMessage(content="hello")))

    assert response[-1] == AssistantMessage(content="done")
    assert ErrorMessage(content="Max retries exceeded") not in response
    assert len(client.requests) == 4


def test_agent_yields_error_message_when_max_turns_exceeded() -> None:
    """Max turn exhaustion yields an error message, then the run ends."""
    agent = _agent(
        client=_ToolLoopClient(),
        max_turns=1,
        permission_manager=_StubManager(tools=[EchoTool()]),
    )
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
    )
    assert next(stream) == ErrorMessage(content="Max turns exceeded")

    with pytest.raises(StopIteration):
        next(stream)


def test_agent_yields_tool_result_before_next_model_request() -> None:
    """Tool results are yielded before the follow-up model request."""
    client = _ToolLoopClient()
    agent = _agent(client=client, permission_manager=_StubManager(tools=[EchoTool()]))

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
    )
    assert len(client.requests) == 1

    assert next(stream) == AssistantMessage(content="done")
    assert len(client.requests) == 2
    assert list(stream) == []


def test_agent_repr_includes_configuration_without_history_contents() -> None:
    """Agent repr exposes debug state without dumping conversation content."""
    agent = _agent(
        client=_ReasoningClient(), max_turns=3, permission_manager=_StubManager()
    )
    agent._add_message_to_history(UserMessage(content="secret"))

    assert repr(agent) == (
        "Agent("
        "client=_ReasoningClient("
        "model_id='fake-model', thinking_effort=None, thinking_summary=None"
        "), "
        "permissions=_StubManager(), "
        "max_turns=3, "
        "max_retries=5, "
        "tools=[]"
        ")"
    )


def test_agent_denied_tool_yields_error_and_skips_execution() -> None:
    """A denied tool call yields a tool error and does not execute."""
    client = _ToolLoopClient()
    manager = _StubManager(tools=[EchoTool()], decisions={"EchoTool": False})
    agent = _agent(client=client, permission_manager=manager)

    response = list(agent.run(UserMessage(content="hello")))

    denial = "Tool call EchoTool denied by user."
    assert response == [
        ToolCallMessage(
            content='EchoTool({"text": "hello"})',
            id="call_1",
            name="EchoTool",
            arguments={"text": "hello"},
        ),
        ToolErrorMessage(content=denial, id="call_1"),
        AssistantMessage(content="done"),
    ]
    assert type(response[1]) is ToolErrorMessage


def test_agent_allowed_tool_with_note_yields_result_then_note() -> None:
    """Approving with a note yields the result and the note as separate messages."""
    client = _ToolLoopClient()
    note = UserMessage(content="use it carefully")
    manager = _StubManager(
        tools=[EchoTool()], decisions={"EchoTool": True}, message=note
    )
    agent = _agent(client=client, permission_manager=manager)

    response = list(agent.run(UserMessage(content="hello")))

    result = ToolResultMessage(content="hello", id="call_1")
    assert response == [
        ToolCallMessage(
            content='EchoTool({"text": "hello"})',
            id="call_1",
            name="EchoTool",
            arguments={"text": "hello"},
        ),
        result,
        note,
        AssistantMessage(content="done"),
    ]
    # both the result and the note are stored in history for the model
    assert result in agent._message_history
    assert note in agent._message_history


def test_agent_allowed_auto_tool_executes() -> None:
    """An allowed tool call executes exactly as without a manager."""
    client = _ToolLoopClient()
    manager = _StubManager(tools=[EchoTool()], decisions={"EchoTool": True})
    agent = _agent(client=client, permission_manager=manager)

    response = list(agent.run(UserMessage(content="hello")))

    assert response == [
        ToolCallMessage(
            content='EchoTool({"text": "hello"})',
            id="call_1",
            name="EchoTool",
            arguments={"text": "hello"},
        ),
        ToolResultMessage(content="hello", id="call_1"),
        AssistantMessage(content="done"),
    ]


def test_agent_exposes_only_manager_tools() -> None:
    """The agent catalog is exactly what the manager builds; nothing else."""
    manager = _StubManager(tools=[])
    agent = _agent(client=_ReasoningClient(), permission_manager=manager)

    assert agent._tools == {}
    assert agent._tool_schemas == []


def test_agent_mixed_calls_allowed_and_denied_keep_history_coherent() -> None:
    """Mixed allowed/denied calls each yield one id-matched result in order."""
    client = _MultiToolLoopClient()
    manager = _StubManager(tools=[EchoTool()], decisions={"a": True, "b": False})
    agent = _agent(client=client, permission_manager=manager)

    response = list(agent.run(UserMessage(content="hello")))

    assert response == [
        ToolCallMessage(
            content='EchoTool({"text": "A"})',
            id="a",
            name="EchoTool",
            arguments={"text": "A"},
        ),
        ToolCallMessage(
            content='EchoTool({"text": "B"})',
            id="b",
            name="EchoTool",
            arguments={"text": "B"},
        ),
        ToolResultMessage(content="A", id="a"),
        ToolErrorMessage(
            content="Tool call EchoTool denied by user.",
            id="b",
        ),
        AssistantMessage(content="done"),
    ]
    results = [msg for msg in response if isinstance(msg, ToolResultMessage)]
    assert sorted(result.id for result in results) == ["a", "b"]


def test_agent_full_permission_manager_builds_all_tools() -> None:
    """A fully-granted permission manager builds every tool."""
    agent = _agent(
        client=_ReasoningClient(),
        permission_manager=_permission_manager(),
    )

    assert set(agent._tools) == {
        "ReadFileTool",
        "WriteFileTool",
        "EditFileTool",
        "GlobTool",
        "GrepTool",
        "BashCommandTool",
        "PowershellCommandTool",
        "SearchWebTool",
        "FetchWebTool",
    }


def test_agent_add_permission_rebuilds_catalog() -> None:
    """Granting a category through the agent rebuilds its tool catalog."""
    manager = _permission_manager()
    manager.remove_permission(PermissionCategory.READ)
    agent = _agent(client=_ReasoningClient(), permission_manager=manager)
    assert "ReadFileTool" not in agent._tools

    agent.add_permission(PermissionCategory.READ)

    assert {"ReadFileTool", "GlobTool", "GrepTool"} <= set(agent._tools)


def test_agent_remove_permission_rebuilds_catalog() -> None:
    """Revoking a category through the agent drops its tools."""
    agent = _agent(
        client=_ReasoningClient(),
        permission_manager=_permission_manager(
            {PermissionCategory.READ: PermissionLevel.ASK}
        ),
    )
    assert set(agent._tools) == {"ReadFileTool", "GlobTool", "GrepTool"}

    agent.remove_permission(PermissionCategory.READ)

    assert agent._tools == {}


def test_agent_set_permission_level_updates_grant_keeping_catalog() -> None:
    """Changing a level via the agent updates the grant, not the catalog."""
    manager = _permission_manager({PermissionCategory.READ: PermissionLevel.ASK})
    agent = _agent(client=_ReasoningClient(), permission_manager=manager)
    before = set(agent._tools)

    agent.set_permission_level(PermissionCategory.READ, PermissionLevel.AUTO)

    assert (
        manager.get_category_permission(PermissionCategory.READ) is PermissionLevel.AUTO
    )
    assert set(agent._tools) == before


def test_agent_repr_renders_tools_via_their_repr() -> None:
    """The agent repr renders each tool through its own repr, not just names."""
    agent = _agent(
        client=_ReasoningClient(), permission_manager=_StubManager(tools=[EchoTool()])
    )

    assert "tools=[EchoTool()]" in repr(agent)


def test_agent_str_pretty_prints_one_tool_per_line() -> None:
    """Str(agent) is multi-line and renders each tool via its repr."""
    agent = _agent(
        client=_ReasoningClient(), permission_manager=_StubManager(tools=[EchoTool()])
    )

    text = str(agent)
    assert text.startswith("Agent(\n")
    assert "\n        EchoTool(),\n" in text
