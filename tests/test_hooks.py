"""Tests for the hook enum and the synchronous registration/dispatch primitive.

These tests exercise ``HookManager`` in isolation, independent of the call sites that
dispatch it in ``Agent``, ``BaseTool``, ``PermissionManager`` and the app (see the
dispatch tests in ``test_agents.py``, ``test_tools.py``, ``test_permissions.py`` and
``test_plugins.py``): registration order, per-hook isolation, per-manager isolation,
exception propagation and the snapshot semantics of ``run``.
"""

from typing import Any

import pytest

from arancio.core.hooks.base import Hook
from arancio.core.hooks.manager import HookManager


def test_the_enum_contains_exactly_the_agreed_names() -> None:
    """No hook is missing, renamed or extra relative to the agreed set."""
    assert {member.value for member in Hook} == {
        "startup",
        "shutdown",
        "agent_start",
        "agent_end",
        "turn_start",
        "turn_end",
        "system_prompt_build",
        "before_model_request",
        "after_model_response",
        "message_received",
        "before_tool_call",
        "after_tool_call",
        "before_permission_check",
        "after_permission_check",
        "error",
    }


def test_event_data_is_written_into_the_context_handlers_receive() -> None:
    """A handler receives the manager's context, holding the event's own objects."""
    manager = HookManager()
    received: list[dict[str, Any]] = []
    sentinel = object()

    manager.register(Hook.AGENT_START, received.append)
    manager.run(Hook.AGENT_START, call=sentinel, name="ReadFileTool")

    assert received == [manager.context]
    assert received[0] is manager.context
    assert manager.context["call"] is sentinel
    assert manager.context["name"] == "ReadFileTool"


def test_every_handler_and_dispatch_share_one_context() -> None:
    """The same dict object reaches every handler of every dispatch."""
    manager = HookManager()
    received: list[dict[str, Any]] = []

    manager.register(Hook.TURN_START, received.append)
    manager.register(Hook.TURN_START, received.append)
    manager.register(Hook.TURN_END, received.append)
    manager.run(Hook.TURN_START, turn=0)
    manager.run(Hook.TURN_END, turn=0)

    assert len(received) == 3
    assert all(context is manager.context for context in received)


def test_event_keys_stay_in_the_context_until_overwritten() -> None:
    """A later dispatch keeps earlier event keys and overwrites only its own."""
    manager = HookManager()

    manager.run(Hook.AFTER_TOOL_CALL, name="ReadFileTool", result="done")
    manager.run(Hook.TURN_START, turn=1)

    assert manager.context == {
        "hook_manager": manager,
        "name": "ReadFileTool",
        "result": "done",
        "turn": 1,
    }


def test_a_new_manager_holds_only_itself_in_its_context() -> None:
    """The manager registers itself; nothing else is there before a dispatch."""
    manager = HookManager()

    assert manager.context == {"hook_manager": manager}


def test_a_value_assigned_by_a_handler_reaches_later_handlers_and_the_caller() -> None:
    """Replacing a key in the shared context is how a handler changes a value."""
    manager = HookManager()
    seen: list[str] = []

    def replace(context: dict[str, Any]) -> None:
        """Append a rule to the prompt."""
        context["system_prompt"] += " Answer in Italian."

    manager.register(Hook.SYSTEM_PROMPT_BUILD, replace)
    manager.register(
        Hook.SYSTEM_PROMPT_BUILD, lambda context: seen.append(context["system_prompt"])
    )
    manager.run(Hook.SYSTEM_PROMPT_BUILD, system_prompt="Be brief.")

    assert seen == ["Be brief. Answer in Italian."]
    assert manager.context["system_prompt"] == "Be brief. Answer in Italian."


def test_handlers_run_in_registration_order() -> None:
    """Dispatch is sequential, not concurrent or reordered."""
    manager = HookManager()
    order: list[int] = []

    manager.register(Hook.TURN_START, lambda context: order.append(1))
    manager.register(Hook.TURN_START, lambda context: order.append(2))
    manager.register(Hook.TURN_START, lambda context: order.append(3))
    manager.run(Hook.TURN_START)

    assert order == [1, 2, 3]


def test_a_handler_only_runs_for_the_hook_it_registered_for() -> None:
    """Registering for one hook does not wire a handler to another."""
    manager = HookManager()
    calls: list[str] = []

    manager.register(Hook.BEFORE_TOOL_CALL, lambda context: calls.append("before"))
    manager.register(Hook.AFTER_TOOL_CALL, lambda context: calls.append("after"))
    manager.run(Hook.BEFORE_TOOL_CALL)

    assert calls == ["before"]


def test_two_managers_do_not_share_handlers() -> None:
    """There is no global registry: each manager is its own namespace."""
    first, second = HookManager(), HookManager()
    calls: list[str] = []

    first.register(Hook.AGENT_START, lambda context: calls.append("first"))
    second.run(Hook.AGENT_START)

    assert calls == []


def test_a_hook_with_no_handlers_does_nothing() -> None:
    """Dispatching an unregistered hook is a silent no-op, not an error."""
    HookManager().run(Hook.AGENT_END)


def test_registering_the_same_handler_twice_runs_it_twice() -> None:
    """Each registration adds one invocation, even for the same callable."""
    manager = HookManager()
    calls: list[int] = []

    def handler(context: dict[str, Any]) -> None:
        """Record one invocation."""
        calls.append(1)

    manager.register(Hook.TURN_END, handler)
    manager.register(Hook.TURN_END, handler)
    manager.run(Hook.TURN_END)

    assert calls == [1, 1]


def test_a_handlers_return_value_is_ignored() -> None:
    """Dispatch has no result-transformation or blocking-result protocol."""
    manager = HookManager()
    manager.register(Hook.MESSAGE_RECEIVED, lambda context: False)

    assert manager.run(Hook.MESSAGE_RECEIVED) == []


def test_an_exception_propagates_and_stops_remaining_handlers() -> None:
    """A handler's exception is not swallowed, and later handlers do not run."""
    manager = HookManager()
    calls: list[str] = []

    def failing(context: dict[str, Any]) -> None:
        """Raise to interrupt dispatch.

        Raises:
            ValueError: always, to simulate a failing handler.
        """
        raise ValueError("boom")

    manager.register(Hook.ERROR, lambda context: calls.append("first"))
    manager.register(Hook.ERROR, failing)
    manager.register(Hook.ERROR, lambda context: calls.append("third"))

    with pytest.raises(ValueError):
        manager.run(Hook.ERROR)

    assert calls == ["first"]


def test_registering_during_dispatch_takes_effect_only_on_the_next_run() -> None:
    """Dispatch iterates a snapshot, so mid-run registration does not run now."""
    manager = HookManager()
    calls: list[str] = []

    def register_another(context: dict[str, Any]) -> None:
        """Register a second handler while the first is still dispatching."""
        manager.register(Hook.ERROR, lambda context: calls.append("late"))
        calls.append("early")

    manager.register(Hook.ERROR, register_another)
    manager.run(Hook.ERROR)

    assert calls == ["early"]

    calls.clear()
    manager.run(Hook.ERROR)

    assert calls == ["early", "late"]


def test_handler_messages_are_returned_in_order() -> None:
    """Messages survive dispatch while unrelated return values are ignored."""
    from arancio.core.messages import ErrorMessage

    manager = HookManager()
    first = ErrorMessage(content="first")
    second = ErrorMessage(content="second")
    manager.register(Hook.TURN_START, lambda context: first)
    manager.register(Hook.TURN_START, lambda context: None)
    manager.register(Hook.TURN_START, lambda context: second)
    assert manager.run(Hook.TURN_START) == [first, second]
