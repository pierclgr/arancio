"""Tests for the agent turn loop.

Everything here runs the real loop against a scripted client and controller. The loop's
contract is mostly about *ordering* and about what does and does not enter model
history, so most assertions are on the sequence of yielded messages and on the history
the next request is built from.
"""

from typing import Any, List

import pytest
from fakes import ScriptedClient, ScriptedController

import arancio.core.agents as agents_module
from arancio.core.agents import Agent
from arancio.core.controllers.responses import Decision
from arancio.core.messages import (
    AssistantChunkMessage,
    AssistantMessage,
    ErrorMessage,
    Message,
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
    UserMessage,
)
from arancio.core.permissions.manager import PermissionManager
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.core.tools.files.read import ReadFileTool
from arancio.core.tools.manager import ToolManager


def _tool_call(name: str = "NoSuchTool", call_id: str = "c1") -> ToolCallMessage:
    """Build a tool call the model is asking for.

    Args:
        name: the tool class name requested.
        call_id: the provider-issued call identifier.

    Returns:
        A tool call message with no arguments.
    """
    return ToolCallMessage(content="", id=call_id, name=name, arguments={})


def _agent(
    permission_manager: PermissionManager,
    turns: List[List[Any]],
    **kwargs: Any,
) -> tuple[Agent, ScriptedClient]:
    """Build an agent over a scripted client.

    Args:
        permission_manager: the manager gating tool calls.
        turns: the per-turn scripts the client replays.
        **kwargs: loop limits forwarded to the agent.

    Returns:
        The agent and the client it is driven by.
    """
    client = ScriptedClient(turns)
    return Agent(client=client, permission_manager=permission_manager, **kwargs), client


def test_the_opening_message_is_remembered_but_not_echoed(
    permission_manager: PermissionManager,
) -> None:
    """The caller already holds what it sent, so yielding it would show it twice."""
    agent, client = _agent(permission_manager, [[AssistantMessage(content="hi")]])

    produced = list(agent.run(UserMessage(content="hello")))

    assert produced == [AssistantMessage(content="hi")]
    assert client.histories[0][0] == UserMessage(content="hello")


def test_prelude_messages_are_both_remembered_and_echoed(
    permission_manager: PermissionManager,
) -> None:
    """Mention results are new to the caller, so unlike the prompt they are yielded."""
    agent, client = _agent(permission_manager, [[AssistantMessage(content="hi")]])
    prelude: List[Message] = [UserMessage(content="file contents")]

    produced = list(agent.run(UserMessage(content="hello"), prelude=prelude))

    assert produced[0] == UserMessage(content="file contents")
    assert client.histories[0][1] == UserMessage(content="file contents")


def test_streaming_chunks_reach_the_user_but_not_the_model(
    permission_manager: PermissionManager,
) -> None:
    """A fragment is shown live, then the finalized message is what is remembered."""
    agent, client = _agent(
        permission_manager,
        [
            [
                AssistantChunkMessage(content="he"),
                AssistantMessage(content="hello"),
                _tool_call(),
            ],
            [AssistantMessage(content="done")],
        ],
    )

    produced = list(agent.run(UserMessage(content="x")))

    assert produced[0] == AssistantChunkMessage(content="he")
    assert produced[1] == AssistantMessage(content="hello")
    assert AssistantMessage(content="hello") in client.histories[1]
    assert AssistantChunkMessage(content="he") not in client.histories[1]


def test_a_text_only_reply_ends_the_loop(
    permission_manager: PermissionManager,
) -> None:
    """No tool calls means the model is finished; no closing message is added."""
    agent, client = _agent(permission_manager, [[AssistantMessage(content="done")]])

    produced = list(agent.run(UserMessage(content="x")))

    assert produced == [AssistantMessage(content="done")]
    assert len(client.requests) == 1


def test_a_tool_call_runs_and_the_loop_takes_another_turn(
    tool_manager: ToolManager, controller: ScriptedController, tmp_path: Any
) -> None:
    """The result is fed back and the model gets a second turn to use it."""
    path = tmp_path / "a.txt"
    path.write_text("alpha\n")
    permission_manager = PermissionManager(
        tool_manager=tool_manager,
        controller=controller,
        permissions={c: PermissionLevel.AUTO for c in PermissionCategory},
    )
    call = ToolCallMessage(
        content="", id="c1", name="ReadFileTool", arguments={"file_path": str(path)}
    )
    agent, client = _agent(
        permission_manager, [[call], [AssistantMessage(content="read it")]]
    )

    produced = list(agent.run(UserMessage(content="x")))

    assert produced[0] is call
    result = produced[1]
    assert isinstance(result, ToolResultMessage)
    assert result.id == "c1"
    assert "alpha" in result.display_text
    assert produced[2] == AssistantMessage(content="read it")
    assert len(client.requests) == 2


def test_an_approval_note_arrives_after_the_result(
    tool_manager: ToolManager, tmp_path: Any
) -> None:
    """The model reports the outcome first, then answers what the user added.

    The wording asks for exactly that order, so the note must not overtake the result it
    is commenting on.
    """
    path = tmp_path / "a.txt"
    path.write_text("alpha\n")
    controller = ScriptedController([(Decision.ALLOW, "also check the tests")])
    permission_manager = PermissionManager(
        tool_manager=tool_manager, controller=controller
    )
    call = ToolCallMessage(
        content="", id="c1", name="ReadFileTool", arguments={"file_path": str(path)}
    )
    agent, _ = _agent(permission_manager, [[call], [AssistantMessage(content="ok")]])

    produced = list(agent.run(UserMessage(content="x")))

    result = produced[1]
    assert isinstance(result, ToolResultMessage)
    assert result.id == "c1"
    note = produced[2]
    assert isinstance(note, UserMessage)
    assert note.display_text == "also check the tests"
    assert "First report tool calling result" in note.content


def test_a_denial_is_reported_to_the_model_and_the_loop_goes_on(
    tool_manager: ToolManager,
) -> None:
    """A refusal is an answer, not a crash: the model gets told and keeps working."""
    controller = ScriptedController([(Decision.DENY, "too risky")])
    permission_manager = PermissionManager(
        tool_manager=tool_manager, controller=controller
    )
    call = ToolCallMessage(
        content="", id="c1", name="ShellCommandTool", arguments={"command": "rm -rf /"}
    )
    agent, client = _agent(
        permission_manager, [[call], [AssistantMessage(content="understood")]]
    )

    produced = list(agent.run(UserMessage(content="x")))

    refusal = produced[1]
    assert isinstance(refusal, ToolErrorMessage)
    assert refusal.content == (
        "Tool call ShellCommandTool denied by user. "
        "Additional information from user: too risky"
    )
    assert refusal in client.histories[1]


def test_a_tool_the_run_does_not_have_is_not_reported_as_a_refusal(
    permission_manager: PermissionManager,
) -> None:
    """A missing tool must not read as the user saying no."""
    agent, _ = _agent(
        permission_manager, [[_tool_call()], [AssistantMessage(content="ok")]]
    )

    produced = list(agent.run(UserMessage(content="x")))

    assert produced[1].content == "Tool NoSuchTool does not exist."


def test_a_turn_failure_is_shown_but_never_sent_back_to_the_model(
    permission_manager: PermissionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error describes this run, so it is surfaced but kept out of context."""
    monkeypatch.setattr(agents_module.time, "sleep", lambda _: None)
    agent, client = _agent(
        permission_manager,
        [[RuntimeError("provider down")], [AssistantMessage(content="recovered")]],
    )

    produced = list(agent.run(UserMessage(content="x")))

    assert isinstance(produced[0], ErrorMessage)
    assert produced[0].content == "Error while executing user request: provider down"
    assert not any(isinstance(m, ErrorMessage) for m in client.histories[1])


def test_consecutive_failures_back_off_and_then_give_up(
    permission_manager: PermissionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each retry waits longer, and the run aborts on the configured attempt."""
    waits: List[float] = []
    monkeypatch.setattr(agents_module.time, "sleep", waits.append)
    agent, _ = _agent(
        permission_manager,
        [[RuntimeError("a")], [RuntimeError("b")], [RuntimeError("c")]],
        max_retries=3,
        retry_delay=1.0,
        retry_delay_multiplier=2.0,
    )

    produced = list(agent.run(UserMessage(content="x")))

    assert waits == [1.0, 2.0]
    assert produced[-1].content == "Max retries exceeded"


def test_a_good_turn_clears_the_backoff(
    permission_manager: PermissionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retries count consecutive failures, so success starts the wait over."""
    waits: List[float] = []
    monkeypatch.setattr(agents_module.time, "sleep", waits.append)
    agent, _ = _agent(
        permission_manager,
        [
            [RuntimeError("a")],
            [_tool_call()],
            [RuntimeError("b")],
            [AssistantMessage(content="done")],
        ],
        retry_delay=1.0,
        retry_delay_multiplier=2.0,
    )

    list(agent.run(UserMessage(content="x")))

    assert waits == [1.0, 1.0]


def test_a_failure_after_delivered_messages_does_not_retry(
    permission_manager: PermissionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retrying would replay work the user already saw, so the run just stops."""
    waits: List[float] = []
    monkeypatch.setattr(agents_module.time, "sleep", waits.append)
    agent, client = _agent(
        permission_manager,
        [[AssistantMessage(content="partial"), RuntimeError("late failure")]],
    )

    produced = list(agent.run(UserMessage(content="x")))

    assert produced[0] == AssistantMessage(content="partial")
    assert isinstance(produced[1], ErrorMessage)
    assert waits == []
    assert len(client.requests) == 1


def test_a_bounded_run_stops_after_its_turn_budget(
    permission_manager: PermissionManager,
) -> None:
    """A model that keeps asking for tools is cut off rather than looping forever."""
    agent, client = _agent(
        permission_manager, [[_tool_call()], [_tool_call()]], max_turns=2
    )

    produced = list(agent.run(UserMessage(content="x")))

    assert produced[-1].content == "Max turns exceeded"
    assert len(client.requests) == 2


def test_restored_history_is_what_the_next_request_is_built_from(
    permission_manager: PermissionManager,
) -> None:
    """A resumed session hands the agent its past, and the model sees it."""
    agent, client = _agent(permission_manager, [[AssistantMessage(content="ok")]])
    agent.restore_history(
        [UserMessage(content="earlier"), AssistantMessage(content="reply")]
    )

    list(agent.run(UserMessage(content="now")))

    contents = [m.content for m in client.histories[0]]
    assert contents == ["earlier", "reply", "now"]


def test_changing_permissions_rebuilds_the_catalog_from_scratch(
    tool_manager: ToolManager, controller: ScriptedController
) -> None:
    """A tool added by hand does not survive the next permission change.

    ``_refresh_tools`` rebuilds the catalog wholesale from the grants, so ``add_tool``
    only holds until something touches permissions. This asserts against the private
    catalog because there is no public way to observe it.
    """
    grants = {category: PermissionLevel.ASK for category in PermissionCategory}
    grants[PermissionCategory.READ] = PermissionLevel.NONE
    permission_manager = PermissionManager(
        tool_manager=tool_manager, controller=controller, permissions=grants
    )
    agent = Agent(client=ScriptedClient(), permission_manager=permission_manager)
    assert "ReadFileTool" not in agent._tools

    agent.add_tool(ReadFileTool())
    assert "ReadFileTool" in agent._tools

    agent.set_permission_level(PermissionCategory.WEB, PermissionLevel.AUTO)

    assert "ReadFileTool" not in agent._tools


def test_a_provided_history_is_copied_not_aliased(
    permission_manager: PermissionManager,
) -> None:
    """The caller's list stays theirs; the agent's turns do not leak into it."""
    history: List[Message] = [UserMessage(content="earlier")]
    client = ScriptedClient([[AssistantMessage(content="ok")]])
    agent = Agent(
        client=client, permission_manager=permission_manager, message_history=history
    )

    list(agent.run(UserMessage(content="now")))

    assert history == [UserMessage(content="earlier")]
