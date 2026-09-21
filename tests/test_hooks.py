"""Tests for the hook enum and the synchronous registration/dispatch primitive.

These tests exercise ``HookManager`` in isolation, independent of the call sites that
dispatch it in ``Agent``, ``BaseTool`` and ``PermissionManager`` (see the dispatch tests
in ``test_agents.py``, ``test_tools.py`` and ``test_permissions.py``): registration
order, per-hook isolation, per-manager isolation, exception propagation and the snapshot
semantics of ``run``.
"""

import pytest

from arancio.core.hooks.manager import HookManager
from arancio.core.hooks.types import Hook


def test_the_enum_contains_exactly_the_agreed_names() -> None:
    """No hook is missing, renamed or extra relative to the agreed set."""
    assert {member.value for member in Hook} == {
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


def test_arguments_forward_unchanged_including_object_identity() -> None:
    """A handler receives the same keyword arguments and the same objects."""
    manager = HookManager()
    received: dict[str, object] = {}
    sentinel = object()

    manager.register(Hook.AGENT_START, lambda **kwargs: received.update(kwargs))
    manager.run(Hook.AGENT_START, call=sentinel, name="ReadFileTool")

    assert received["call"] is sentinel
    assert received["name"] == "ReadFileTool"


def test_handlers_run_in_registration_order() -> None:
    """Dispatch is sequential, not concurrent or reordered."""
    manager = HookManager()
    order: list[int] = []

    manager.register(Hook.TURN_START, lambda: order.append(1))
    manager.register(Hook.TURN_START, lambda: order.append(2))
    manager.register(Hook.TURN_START, lambda: order.append(3))
    manager.run(Hook.TURN_START)

    assert order == [1, 2, 3]


def test_a_handler_only_runs_for_the_hook_it_registered_for() -> None:
    """Registering for one hook does not wire a handler to another."""
    manager = HookManager()
    calls: list[str] = []

    manager.register(Hook.BEFORE_TOOL_CALL, lambda: calls.append("before"))
    manager.register(Hook.AFTER_TOOL_CALL, lambda: calls.append("after"))
    manager.run(Hook.BEFORE_TOOL_CALL)

    assert calls == ["before"]


def test_two_managers_do_not_share_handlers() -> None:
    """There is no global registry: each manager is its own namespace."""
    first, second = HookManager(), HookManager()
    calls: list[str] = []

    first.register(Hook.AGENT_START, lambda: calls.append("first"))
    second.run(Hook.AGENT_START)

    assert calls == []


def test_a_hook_with_no_handlers_does_nothing() -> None:
    """Dispatching an unregistered hook is a silent no-op, not an error."""
    HookManager().run(Hook.AGENT_END)


def test_registering_the_same_handler_twice_runs_it_twice() -> None:
    """Each registration adds one invocation, even for the same callable."""
    manager = HookManager()
    calls: list[int] = []

    def handler() -> None:
        """Record one invocation."""
        calls.append(1)

    manager.register(Hook.TURN_END, handler)
    manager.register(Hook.TURN_END, handler)
    manager.run(Hook.TURN_END)

    assert calls == [1, 1]


def test_a_handlers_return_value_is_ignored() -> None:
    """Dispatch has no result-transformation or blocking-result protocol."""
    manager = HookManager()
    manager.register(Hook.MESSAGE_RECEIVED, lambda: False)

    assert manager.run(Hook.MESSAGE_RECEIVED) is None


def test_an_exception_propagates_and_stops_remaining_handlers() -> None:
    """A handler's exception is not swallowed, and later handlers do not run."""
    manager = HookManager()
    calls: list[str] = []

    def failing() -> None:
        """Raise to interrupt dispatch.

        Raises:
            ValueError: always, to simulate a failing handler.
        """
        raise ValueError("boom")

    manager.register(Hook.ERROR, lambda: calls.append("first"))
    manager.register(Hook.ERROR, failing)
    manager.register(Hook.ERROR, lambda: calls.append("third"))

    with pytest.raises(ValueError):
        manager.run(Hook.ERROR)

    assert calls == ["first"]


def test_registering_during_dispatch_takes_effect_only_on_the_next_run() -> None:
    """Dispatch iterates a snapshot, so mid-run registration does not run now."""
    manager = HookManager()
    calls: list[str] = []

    def register_another() -> None:
        """Register a second handler while the first is still dispatching."""
        manager.register(Hook.ERROR, lambda: calls.append("late"))
        calls.append("early")

    manager.register(Hook.ERROR, register_another)
    manager.run(Hook.ERROR)

    assert calls == ["early"]

    calls.clear()
    manager.run(Hook.ERROR)

    assert calls == ["early", "late"]
