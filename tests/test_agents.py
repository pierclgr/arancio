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
from arancio.core.hooks.manager import HookManager
from arancio.core.hooks.types import Hook
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
        **kwargs: loop limits and/or a ``hook_manager`` forwarded to the
            agent; a private, empty hook manager is used when none is
            given.

    Returns:
        The agent and the client it is driven by.
    """
    client = ScriptedClient(turns)
    kwargs.setdefault("hook_manager", HookManager())
    return Agent(client=client, permission_manager=permission_manager, **kwargs), client


def test_the_opening_message_is_remembered_but_not_echoed(
    permission_manager: PermissionManager,
) -> None:
    """The caller already holds what it sent, so yielding it would show it twice."""
    agent, client = _agent(permission_manager, [[AssistantMessage(content="hi")]])

    produced = list(agent(UserMessage(content="hello")))

    assert produced == [AssistantMessage(content="hi")]
    assert client.histories[0][0] == UserMessage(content="hello")


def test_prelude_messages_are_both_remembered_and_echoed(
    permission_manager: PermissionManager,
) -> None:
    """Mention results are new to the caller, so unlike the prompt they are yielded."""
    agent, client = _agent(permission_manager, [[AssistantMessage(content="hi")]])
    prelude: List[Message] = [UserMessage(content="file contents")]

    produced = list(agent(UserMessage(content="hello"), prelude=prelude))

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

    produced = list(agent(UserMessage(content="x")))

    assert produced[0] == AssistantChunkMessage(content="he")
    assert produced[1] == AssistantMessage(content="hello")
    assert AssistantMessage(content="hello") in client.histories[1]
    assert AssistantChunkMessage(content="he") not in client.histories[1]


def test_a_text_only_reply_ends_the_loop(
    permission_manager: PermissionManager,
) -> None:
    """No tool calls means the model is finished; no closing message is added."""
    agent, client = _agent(permission_manager, [[AssistantMessage(content="done")]])

    produced = list(agent(UserMessage(content="x")))

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
        hook_manager=HookManager(),
        permissions={c: PermissionLevel.AUTO for c in PermissionCategory},
    )
    call = ToolCallMessage(
        content="", id="c1", name="ReadFileTool", arguments={"file_path": str(path)}
    )
    agent, client = _agent(
        permission_manager, [[call], [AssistantMessage(content="read it")]]
    )

    produced = list(agent(UserMessage(content="x")))

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
        tool_manager=tool_manager, controller=controller, hook_manager=HookManager()
    )
    call = ToolCallMessage(
        content="", id="c1", name="ReadFileTool", arguments={"file_path": str(path)}
    )
    agent, _ = _agent(permission_manager, [[call], [AssistantMessage(content="ok")]])

    produced = list(agent(UserMessage(content="x")))

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
        tool_manager=tool_manager, controller=controller, hook_manager=HookManager()
    )
    call = ToolCallMessage(
        content="", id="c1", name="ShellCommandTool", arguments={"command": "rm -rf /"}
    )
    agent, client = _agent(
        permission_manager, [[call], [AssistantMessage(content="understood")]]
    )

    produced = list(agent(UserMessage(content="x")))

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

    produced = list(agent(UserMessage(content="x")))

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

    produced = list(agent(UserMessage(content="x")))

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

    produced = list(agent(UserMessage(content="x")))

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

    list(agent(UserMessage(content="x")))

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

    produced = list(agent(UserMessage(content="x")))

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

    produced = list(agent(UserMessage(content="x")))

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

    list(agent(UserMessage(content="now")))

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
        tool_manager=tool_manager,
        controller=controller,
        hook_manager=HookManager(),
        permissions=grants,
    )
    agent = Agent(
        client=ScriptedClient(),
        permission_manager=permission_manager,
        hook_manager=HookManager(),
    )
    assert "ReadFileTool" not in agent._tools

    agent.add_tool(ReadFileTool(hook_manager=HookManager()))
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
        client=client,
        permission_manager=permission_manager,
        hook_manager=HookManager(),
        message_history=history,
    )

    list(agent(UserMessage(content="now")))

    assert history == [UserMessage(content="earlier")]


def _record(manager: HookManager, *hooks: Hook) -> List[Any]:
    """Register a handler on each hook that appends its dispatch to a list.

    Args:
        manager: the hook manager to register against.
        *hooks: the hooks to record.

    Returns:
        The list handlers append ``(hook, kwargs)`` to, in dispatch order.
    """
    events: List[Any] = []
    for hook in hooks:
        manager.register(
            hook, lambda hook=hook, **kwargs: events.append((hook, kwargs))
        )
    return events


def test_agent_start_dispatches_before_the_opening_message_is_recorded(
    permission_manager: PermissionManager,
) -> None:
    """The dispatch happens before the message is appended to history."""
    hooks = HookManager()
    snapshots: List[Any] = []
    hooks.register(
        Hook.AGENT_START,
        lambda message: snapshots.append((message, list(agent._message_history))),
    )
    agent, _ = _agent(
        permission_manager, [[AssistantMessage(content="hi")]], hook_manager=hooks
    )

    list(agent(UserMessage(content="hello")))

    assert len(snapshots) == 1
    message, history_at_dispatch = snapshots[0]
    assert message == UserMessage(content="hello")
    assert history_at_dispatch == []


def test_turn_start_and_turn_end_fire_once_per_turn_in_order(
    permission_manager: PermissionManager,
) -> None:
    """Both hooks fire exactly once per turn, start before end."""
    hooks = HookManager()
    events = _record(hooks, Hook.TURN_START, Hook.TURN_END)
    agent, _ = _agent(
        permission_manager,
        [[_tool_call()], [AssistantMessage(content="done")]],
        hook_manager=hooks,
    )

    list(agent(UserMessage(content="x")))

    assert [hook for hook, _ in events] == [
        Hook.TURN_START,
        Hook.TURN_END,
        Hook.TURN_START,
        Hook.TURN_END,
    ]
    assert events[0][1]["turn"] == 0
    assert events[2][1]["turn"] == 1


def test_turn_end_still_fires_when_the_turn_raises(
    permission_manager: PermissionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed turn still gets its ``turn_end``, from the ``finally``."""
    monkeypatch.setattr(agents_module.time, "sleep", lambda _: None)
    hooks = HookManager()
    events = _record(hooks, Hook.TURN_START, Hook.TURN_END)
    agent, _ = _agent(
        permission_manager,
        [[RuntimeError("provider down")], [AssistantMessage(content="recovered")]],
        hook_manager=hooks,
    )

    list(agent(UserMessage(content="x")))

    assert [hook for hook, _ in events] == [
        Hook.TURN_START,
        Hook.TURN_END,
        Hook.TURN_START,
        Hook.TURN_END,
    ]


def test_system_prompt_build_fires_with_the_rendered_prompt(
    permission_manager: PermissionManager,
) -> None:
    """The dispatched kwarg is the same string the request was built with."""
    hooks = HookManager()
    events = _record(hooks, Hook.SYSTEM_PROMPT_BUILD)
    agent, client = _agent(
        permission_manager, [[AssistantMessage(content="hi")]], hook_manager=hooks
    )

    list(agent(UserMessage(content="x")))

    assert len(events) == 1
    assert events[0][1]["system_prompt"] == client.requests[0].system_prompt
    assert events[0][1]["system_prompt"] != ""


def test_before_model_request_and_after_model_response_carry_the_same_request(
    tool_manager: ToolManager, controller: ScriptedController, tmp_path: Any
) -> None:
    """Both dispatches see the exact same request object the client received."""
    path = tmp_path / "a.txt"
    path.write_text("alpha\n")
    permission_manager = PermissionManager(
        tool_manager=tool_manager,
        controller=controller,
        hook_manager=HookManager(),
        permissions={c: PermissionLevel.AUTO for c in PermissionCategory},
    )
    call = ToolCallMessage(
        content="", id="c1", name="ReadFileTool", arguments={"file_path": str(path)}
    )
    hooks = HookManager()
    events = _record(hooks, Hook.BEFORE_MODEL_REQUEST, Hook.AFTER_MODEL_RESPONSE)
    agent, client = _agent(
        permission_manager,
        [[call], [AssistantMessage(content="read it")]],
        hook_manager=hooks,
    )

    list(agent(UserMessage(content="x")))

    before, after = events[0][1], events[1][1]
    assert before["request"] is client.requests[0]
    assert after["request"] is client.requests[0]
    assert after["tool_calls"] == [call]


def test_message_received_fires_only_for_finalized_messages_not_chunks(
    permission_manager: PermissionManager,
) -> None:
    """A streamed chunk does not itself trigger ``message_received``."""
    hooks = HookManager()
    events = _record(hooks, Hook.MESSAGE_RECEIVED)
    agent, _ = _agent(
        permission_manager,
        [[AssistantChunkMessage(content="he"), AssistantMessage(content="hello")]],
        hook_manager=hooks,
    )

    list(agent(UserMessage(content="x")))

    assert len(events) == 1
    assert events[0][1]["response_message"] == AssistantMessage(content="hello")


def test_agent_end_fires_on_a_natural_stop_and_error_does_not(
    permission_manager: PermissionManager,
) -> None:
    """A plain-text reply is the only outcome that fires ``agent_end``."""
    hooks = HookManager()
    events = _record(hooks, Hook.AGENT_END, Hook.ERROR)
    agent, _ = _agent(
        permission_manager, [[AssistantMessage(content="done")]], hook_manager=hooks
    )

    list(agent(UserMessage(content="x")))

    assert [hook for hook, _ in events] == [Hook.AGENT_END]


def test_error_fires_with_source_model_on_every_failed_turn_including_retried_ones(
    permission_manager: PermissionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every failed attempt dispatches ``error`` with ``source="model"``."""
    monkeypatch.setattr(agents_module.time, "sleep", lambda _: None)
    hooks = HookManager()
    events = _record(hooks, Hook.ERROR)
    errors = [RuntimeError("a"), RuntimeError("b"), RuntimeError("c")]
    agent, _ = _agent(
        permission_manager,
        [[errors[0]], [errors[1]], [errors[2]]],
        max_retries=3,
        hook_manager=hooks,
    )

    list(agent(UserMessage(content="x")))

    model_errors = [kw["error"] for _, kw in events if kw["source"] == "model"]
    assert model_errors == errors


def test_error_with_source_model_carries_no_request_when_build_request_fails(
    permission_manager: PermissionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``request`` is ``None`` in the dispatch when ``build_request`` itself raised."""
    monkeypatch.setattr(agents_module.time, "sleep", lambda _: None)
    hooks = HookManager()
    events = _record(hooks, Hook.ERROR)
    agent, client = _agent(
        permission_manager, [[AssistantMessage(content="ok")]], hook_manager=hooks
    )

    def _raise(**kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(client, "build_request", _raise)

    list(agent(UserMessage(content="x")))

    assert events[0][1]["source"] == "model"
    assert events[0][1]["request"] is None


def test_error_with_source_agent_fires_once_when_retries_are_exhausted(
    permission_manager: PermissionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``source="agent"`` dispatch fires only on the final, unrecovered failure."""
    monkeypatch.setattr(agents_module.time, "sleep", lambda _: None)
    hooks = HookManager()
    events = _record(hooks, Hook.ERROR)
    agent, _ = _agent(
        permission_manager,
        [[RuntimeError("a")], [RuntimeError("b")], [RuntimeError("c")]],
        max_retries=3,
        hook_manager=hooks,
    )

    list(agent(UserMessage(content="x")))

    agent_errors = [kw for _, kw in events if kw["source"] == "agent"]
    assert len(agent_errors) == 1
    assert str(agent_errors[0]["error"]) == "c"


def test_error_with_source_agent_carries_no_error_when_max_turns_is_exceeded(
    permission_manager: PermissionManager,
) -> None:
    """Loop exhaustion (not an exception) still dispatches a ``source="agent"`` one."""
    hooks = HookManager()
    events = _record(hooks, Hook.ERROR)
    agent, _ = _agent(
        permission_manager,
        [[_tool_call()], [_tool_call()]],
        max_turns=2,
        hook_manager=hooks,
    )

    list(agent(UserMessage(content="x")))

    assert len(events) == 1
    assert events[0][1]["source"] == "agent"
    assert events[0][1]["error"] is None


def test_error_fires_with_source_model_then_agent_after_delivered_messages(
    permission_manager: PermissionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``received_finalized`` early return dispatches both sources, in order."""
    monkeypatch.setattr(agents_module.time, "sleep", lambda _: None)
    hooks = HookManager()
    events = _record(hooks, Hook.ERROR)
    agent, _ = _agent(
        permission_manager,
        [[AssistantMessage(content="partial"), RuntimeError("late failure")]],
        hook_manager=hooks,
    )

    list(agent(UserMessage(content="x")))

    assert [kw["source"] for _, kw in events] == ["model", "agent"]


def test_a_hook_handler_exception_during_a_tool_call_surfaces_as_source_model(
    controller: ScriptedController, tmp_path: Any
) -> None:
    """A broken ``before_tool_call`` handler is not caught by ``BaseTool.call``.

    It propagates to the loop's own ``except``, the same one that catches a genuine
    model-call failure, and is dispatched the same way: ``error`` with
    ``source="model"``, never ``source="tool"`` since ``_call`` itself never ran.
    """
    path = tmp_path / "a.txt"
    path.write_text("alpha\n")
    hooks = HookManager()
    events = _record(hooks, Hook.ERROR)
    hooks.register(
        Hook.BEFORE_TOOL_CALL, lambda **_: (_ for _ in ()).throw(RuntimeError)
    )
    tool_manager = ToolManager(web_summary_client=ScriptedClient(), hook_manager=hooks)
    permission_manager = PermissionManager(
        tool_manager=tool_manager,
        controller=controller,
        permissions={c: PermissionLevel.AUTO for c in PermissionCategory},
        hook_manager=hooks,
    )
    call = ToolCallMessage(
        content="", id="c1", name="ReadFileTool", arguments={"file_path": str(path)}
    )
    agent, _ = _agent(permission_manager, [[call]], max_retries=1, hook_manager=hooks)

    produced = list(agent(UserMessage(content="x")))

    assert any(
        isinstance(m, ErrorMessage)
        and m.content.startswith("Error while executing user request:")
        for m in produced
    )
    assert [kw["source"] for _, kw in events] == ["model", "agent"]


def test_a_shared_hook_manager_sees_every_component_in_order(
    controller: ScriptedController, tmp_path: Any
) -> None:
    """One shared manager observes agent, permission and tool dispatch, in order."""
    path = tmp_path / "a.txt"
    path.write_text("alpha\n")
    hooks = HookManager()
    events = _record(
        hooks,
        Hook.AGENT_START,
        Hook.TURN_START,
        Hook.SYSTEM_PROMPT_BUILD,
        Hook.BEFORE_MODEL_REQUEST,
        Hook.MESSAGE_RECEIVED,
        Hook.AFTER_MODEL_RESPONSE,
        Hook.BEFORE_PERMISSION_CHECK,
        Hook.AFTER_PERMISSION_CHECK,
        Hook.BEFORE_TOOL_CALL,
        Hook.AFTER_TOOL_CALL,
        Hook.AGENT_END,
        Hook.TURN_END,
    )
    call = ToolCallMessage(
        content="", id="c1", name="ReadFileTool", arguments={"file_path": str(path)}
    )
    client = ScriptedClient([[call], [AssistantMessage(content="read it")]])
    tool_manager = ToolManager(web_summary_client=client, hook_manager=hooks)
    permission_manager = PermissionManager(
        tool_manager=tool_manager,
        controller=controller,
        permissions={c: PermissionLevel.AUTO for c in PermissionCategory},
        hook_manager=hooks,
    )
    agent = Agent(
        client=client, permission_manager=permission_manager, hook_manager=hooks
    )

    list(agent(UserMessage(content="x")))

    assert [hook for hook, _ in events] == [
        Hook.AGENT_START,
        Hook.TURN_START,
        Hook.SYSTEM_PROMPT_BUILD,
        Hook.BEFORE_MODEL_REQUEST,
        Hook.MESSAGE_RECEIVED,
        Hook.AFTER_MODEL_RESPONSE,
        Hook.BEFORE_PERMISSION_CHECK,
        Hook.AFTER_PERMISSION_CHECK,
        Hook.BEFORE_TOOL_CALL,
        Hook.AFTER_TOOL_CALL,
        Hook.TURN_END,
        Hook.TURN_START,
        Hook.SYSTEM_PROMPT_BUILD,
        Hook.BEFORE_MODEL_REQUEST,
        Hook.MESSAGE_RECEIVED,
        Hook.AFTER_MODEL_RESPONSE,
        Hook.AGENT_END,
        Hook.TURN_END,
    ]
