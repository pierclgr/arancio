"""Tests for permission grants, the tool catalog they build and the ask path.

A category at ``NONE`` is not a call that gets refused — it is a tool that is never
built, so the model is never told it exists. That distinction is what these tests pin
down, alongside the two ways a decision reaches the user.
"""

import pytest
from fakes import ScriptedController

from arancio.core.controllers.requests import PermissionRequest
from arancio.core.controllers.responses import Decision
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
    **levels: PermissionLevel,
) -> PermissionManager:
    """Build a permission manager with named categories overridden.

    Args:
        tool_manager: the manager that builds granted tools.
        controller: the controller resolving ``ASK`` calls.
        **levels: category name (lowercase) to level, overriding ``ASK``.

    Returns:
        A permission manager over a complete grant map.
    """
    grants = {category: PermissionLevel.ASK for category in PermissionCategory}
    for name, level in levels.items():
        grants[PermissionCategory[name.upper()]] = level
    return PermissionManager(
        tool_manager=tool_manager, controller=controller, permissions=grants
    )


def test_a_manager_without_grants_asks_for_everything(
    permission_manager: PermissionManager,
) -> None:
    """Omitting the grants means every category is at ``ASK``, never absent."""
    assert permission_manager.validate(_call()).outcome is PermissionOutcome.ALLOWED


def test_an_auto_category_never_reaches_the_user(
    tool_manager: ToolManager, controller: ScriptedController
) -> None:
    """An ``AUTO`` grant is the point where the controller is skipped entirely."""
    manager = _manager(tool_manager, controller, read=PermissionLevel.AUTO)

    decision = manager.validate(_call())

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

    decision = manager.validate(_call())

    assert decision.outcome is PermissionOutcome.DENIED
    assert decision.note == "not that file"


def test_an_empty_note_is_normalized_away(tool_manager: ToolManager) -> None:
    """A blank message is no message, so the agent does not word an empty note."""
    controller = ScriptedController([(Decision.ALLOW, "")])
    manager = _manager(tool_manager, controller)

    assert manager.validate(_call()).note is None


def test_a_tool_in_a_revoked_category_is_unavailable(
    tool_manager: ToolManager, controller: ScriptedController
) -> None:
    """``NONE`` reads as "no such tool", which is what the model is told."""
    manager = _manager(tool_manager, controller, read=PermissionLevel.NONE)

    decision = manager.validate(_call())

    assert decision.outcome is PermissionOutcome.UNAVAILABLE
    assert controller.requests == []


def test_an_unknown_tool_name_is_unavailable(
    permission_manager: PermissionManager, controller: ScriptedController
) -> None:
    """A tool that does not exist resolves without asking anyone."""
    decision = permission_manager.validate(_call("NoSuchTool"))

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
