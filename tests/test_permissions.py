"""Tests for permission grants, the tool catalog they build and the ask path.

A category at ``NONE`` is not a call that gets refused — it is a tool that is never
built, so the model is never told it exists. That distinction is what these tests pin
down, alongside the two ways a decision reaches the user.
"""

import pytest
from fakes import ScriptedController

from arancio.core.controllers.requests import PermissionRequest
from arancio.core.controllers.responses import Decision
from arancio.core.hooks.manager import HookManager
from arancio.core.hooks.types import Hook
from arancio.core.messages import ToolCallMessage
from arancio.core.permissions.manager import PermissionManager
from arancio.core.permissions.types import (
    PermissionCategory,
    PermissionLevel,
    PermissionOutcome,
)
from arancio.core.tools.manager import ToolManager


def _call(name: str = "ReadFileTool") -> ToolCallMessage:
    """Build a tool call for the named tool.

    Args:
        name: the tool class name the model is asking for.

    Returns:
        A tool call message with no arguments.
    """
    return ToolCallMessage(content="", id="c1", name=name, arguments={})


def _manager(
    tool_manager: ToolManager,
    controller: ScriptedController,
    hook_manager: HookManager | None = None,
    **levels: PermissionLevel,
) -> PermissionManager:
    """Build a permission manager with named categories overridden.

    Args:
        tool_manager: the manager that builds granted tools.
        controller: the controller resolving ``ASK`` calls.
        hook_manager: the hook manager ``validate`` dispatches through.
        **levels: category name (lowercase) to level, overriding ``ASK``.

    Returns:
        A permission manager over a complete grant map.
    """
    grants = {category: PermissionLevel.ASK for category in PermissionCategory}
    for name, level in levels.items():
        grants[PermissionCategory[name.upper()]] = level
    if hook_manager is None:
        hook_manager = HookManager()
    return PermissionManager(
        tool_manager=tool_manager,
        controller=controller,
        permissions=grants,
        hook_manager=hook_manager,
    )


def test_a_manager_without_grants_asks_for_everything(
    permission_manager: PermissionManager,
) -> None:
    """Omitting the grants means every category is at ``ASK``, never absent."""
    assert permission_manager.validate(_call())[0].outcome is PermissionOutcome.ALLOWED


def test_an_auto_category_never_reaches_the_user(
    tool_manager: ToolManager, controller: ScriptedController
) -> None:
    """An ``AUTO`` grant is the point where the controller is skipped entirely."""
    manager = _manager(tool_manager, controller, read=PermissionLevel.AUTO)

    decision, hook_messages = manager.validate(_call())

    assert hook_messages == []
    assert decision.outcome is PermissionOutcome.ALLOWED
    assert controller.requests == []


def test_an_ask_category_puts_the_call_to_the_user(
    tool_manager: ToolManager, controller: ScriptedController
) -> None:
    """The user sees the actual call they are approving."""
    manager = _manager(tool_manager, controller)

    manager.validate(_call())

    assert len(controller.requests) == 1
    request = controller.requests[0]
    assert isinstance(request, PermissionRequest)
    assert request.call.name == "ReadFileTool"


def test_a_denial_carries_the_user_note(tool_manager: ToolManager) -> None:
    """The reason the user typed travels with the decision to the agent."""
    controller = ScriptedController([(Decision.DENY, "not that file")])
    manager = _manager(tool_manager, controller)

    decision, hook_messages = manager.validate(_call())

    assert hook_messages == []
    assert decision.outcome is PermissionOutcome.DENIED
    assert decision.note == "not that file"


def test_an_empty_note_is_normalized_away(tool_manager: ToolManager) -> None:
    """A blank message is no message, so the agent does not word an empty note."""
    controller = ScriptedController([(Decision.ALLOW, "")])
    manager = _manager(tool_manager, controller)

    assert manager.validate(_call())[0].note is None


def test_a_tool_in_a_revoked_category_is_unavailable(
    tool_manager: ToolManager, controller: ScriptedController
) -> None:
    """``NONE`` reads as "no such tool", which is what the model is told."""
    manager = _manager(tool_manager, controller, read=PermissionLevel.NONE)

    decision, hook_messages = manager.validate(_call())

    assert hook_messages == []
    assert decision.outcome is PermissionOutcome.UNAVAILABLE
    assert controller.requests == []


def test_an_unknown_tool_name_is_unavailable(
    permission_manager: PermissionManager, controller: ScriptedController
) -> None:
    """A tool that does not exist resolves without asking anyone."""
    decision, hook_messages = permission_manager.validate(_call("NoSuchTool"))

    assert hook_messages == []
    assert decision.outcome is PermissionOutcome.UNAVAILABLE
    assert controller.requests == []


def test_a_revoked_category_builds_none_of_its_tools(
    tool_manager: ToolManager, controller: ScriptedController
) -> None:
    """The tools are not merely refused at call time — they are never created."""
    manager = _manager(tool_manager, controller, write=PermissionLevel.NONE)

    names = {type(tool).__name__ for tool in manager.allowed_tools}

    assert "WriteFileTool" not in names
    assert "EditFileTool" not in names
    assert "ReadFileTool" in names


def test_every_granted_category_contributes_its_tools(
    permission_manager: PermissionManager,
) -> None:
    """All four categories at ``ASK`` build the complete catalog.

    Asserted as a set: a category's tools are a frozenset, so their order is
    not stable between runs.
    """
    names = {type(tool).__name__ for tool in permission_manager.allowed_tools}

    assert names == {
        "ReadFileTool",
        "WriteFileTool",
        "EditFileTool",
        "SearchWebTool",
        "FetchWebTool",
        "ShellCommandTool",
    }


def test_granting_a_category_twice_is_refused(
    tool_manager: ToolManager, controller: ScriptedController
) -> None:
    """Re-granting is a caller mistake, not a silent no-op."""
    manager = _manager(tool_manager, controller, web=PermissionLevel.AUTO)

    with pytest.raises(ValueError):
        manager.add_permission(PermissionCategory.WEB)


def test_revoking_an_ungranted_category_is_refused(
    tool_manager: ToolManager, controller: ScriptedController
) -> None:
    """Revoking what was never granted is likewise surfaced."""
    manager = _manager(tool_manager, controller, web=PermissionLevel.NONE)

    with pytest.raises(ValueError):
        manager.remove_permission(PermissionCategory.WEB)


def test_a_category_maps_to_tool_classes_not_names() -> None:
    """The mapping is by class, so renaming a tool cannot silently ungate it."""
    assert PermissionCategory.for_tool("WriteFileTool") is PermissionCategory.WRITE
    assert PermissionCategory.for_tool("NoSuchTool") is None


def _record(manager: HookManager, *hooks: Hook) -> list:
    """Register a handler on each hook that appends its dispatch to a list.

    Args:
        manager: the hook manager to register against.
        *hooks: the hooks to record.

    Returns:
        The list handlers append ``(hook, kwargs)`` to, in dispatch order.
    """
    events: list = []
    for hook in hooks:
        manager.register(
            hook, lambda hook=hook, **kwargs: events.append((hook, kwargs))
        )
    return events


def test_validate_dispatches_before_and_after_around_an_auto_allow(
    tool_manager: ToolManager, controller: ScriptedController
) -> None:
    """A clean auto-allow fires exactly the before/after pair, in order."""
    hooks = HookManager()
    events = _record(hooks, Hook.BEFORE_PERMISSION_CHECK, Hook.AFTER_PERMISSION_CHECK)
    manager = _manager(tool_manager, controller, hooks, read=PermissionLevel.AUTO)
    call = _call()

    manager.validate(call)

    assert [hook for hook, _ in events] == [
        Hook.BEFORE_PERMISSION_CHECK,
        Hook.AFTER_PERMISSION_CHECK,
    ]
    assert events[0][1]["call"] is call
    assert events[1][1]["decision"].outcome is PermissionOutcome.ALLOWED


def test_validate_dispatches_exactly_one_pair_for_an_unavailable_call(
    tool_manager: ToolManager, controller: ScriptedController
) -> None:
    """Only ``_resolve``'s unavailable branch runs, not all three."""
    hooks = HookManager()
    events = _record(hooks, Hook.BEFORE_PERMISSION_CHECK, Hook.AFTER_PERMISSION_CHECK)
    manager = _manager(tool_manager, controller, hooks)

    manager.validate(_call("NoSuchTool"))

    assert len(events) == 2


def test_validate_dispatches_around_an_ask_denial_carrying_the_note(
    tool_manager: ToolManager,
) -> None:
    """The denial note the user typed reaches the ``after`` dispatch too."""
    hooks = HookManager()
    events = _record(hooks, Hook.AFTER_PERMISSION_CHECK)
    controller = ScriptedController([(Decision.DENY, "not that file")])
    manager = _manager(tool_manager, controller, hooks)

    manager.validate(_call())

    assert events[0][1]["decision"].note == "not that file"


@pytest.mark.parametrize(
    "level, answer, outcome",
    [
        (PermissionLevel.AUTO, Decision.ALLOW, PermissionOutcome.ALLOWED),
        (PermissionLevel.ASK, Decision.ALLOW, PermissionOutcome.ALLOWED),
        (PermissionLevel.ASK, Decision.DENY, PermissionOutcome.DENIED),
        (PermissionLevel.NONE, Decision.ALLOW, PermissionOutcome.UNAVAILABLE),
    ],
)
def test_permission_hook_messages_preserve_decision(
    tool_manager: ToolManager,
    level: PermissionLevel,
    answer: Decision,
    outcome: PermissionOutcome,
) -> None:
    """Before and after messages do not replace any permission outcome or note."""
    from arancio.core.messages import ErrorMessage

    hooks = HookManager()
    before = ErrorMessage(content="before")
    after = ErrorMessage(content="after")
    hooks.register(Hook.BEFORE_PERMISSION_CHECK, lambda **_: before)
    hooks.register(Hook.AFTER_PERMISSION_CHECK, lambda **_: after)
    manager = _manager(
        tool_manager,
        ScriptedController([(answer, "note")]),
        hook_manager=hooks,
        read=level,
    )
    decision, messages = manager.validate(_call())
    assert decision.outcome is outcome
    assert decision.note == ("note" if level is PermissionLevel.ASK else None)
    assert messages == [before, after]
