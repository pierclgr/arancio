"""Tests for the permission system: categories, grants and the manager."""

import pytest

from arancio.core.clients.litellm import LiteLLMClient
from arancio.core.controllers.requests import BaseControllerRequest, PermissionRequest
from arancio.core.controllers.responses import (
    BaseControllerResponse,
    Decision,
    PermissionResponse,
)
from arancio.core.messages import ToolCallMessage, ToolErrorMessage, UserMessage
from arancio.core.permissions.manager import PermissionManager
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.core.tools.manager import ToolManager


class _FakeController:
    """Controller stub returning a preset response and recording requests.

    Attributes:
        requests: the requests passed to :meth:`request`, in order.
    """

    def __init__(self, response: PermissionResponse | None = None) -> None:
        """Store the canned response (``None`` means ``request`` must not run).

        Args:
            response: the response to return; when ``None`` any call to
                :meth:`request` fails the test.
        """
        self._response = response
        self.requests: list[BaseControllerRequest] = []

    def request(self, request: BaseControllerRequest) -> BaseControllerResponse:
        """Record the request and return the canned response.

        Args:
            request: the request to record.

        Returns:
            The canned response supplied at construction.

        Raises:
            AssertionError: when no response was configured.
        """
        self.requests.append(request)
        if self._response is None:
            raise AssertionError("controller.request must not be called")
        return self._response


def _manager(
    permissions: dict[PermissionCategory, PermissionLevel] | None = None,
    controller: _FakeController | None = None,
) -> PermissionManager:
    """Build a permission manager backed by a real tool manager.

    Args:
        permissions: optional category-to-level grants forwarded to the
            manager.
        controller: optional controller stub; defaults to one that fails if
            asked (so AUTO/absent paths assert no prompt happens).

    Returns:
        A :class:`PermissionManager` whose tool manager carries a summary
        client.
    """
    summary_client = LiteLLMClient(
        model_id="ollama_chat/deepseek-v4-flash:cloud", stream=False
    )
    return PermissionManager(
        ToolManager(web_summary_client=summary_client),
        controller or _FakeController(),
        permissions,
    )


def _call(name: str) -> ToolCallMessage:
    """Build a tool call message for the named tool.

    Args:
        name: the tool class name the call targets.

    Returns:
        A tool call message with empty arguments.
    """
    return ToolCallMessage(content="", id="call_1", name=name, arguments={})


def test_for_tool_reverse_lookup() -> None:
    """for_tool resolves known tool names and returns None for unknown ones."""
    assert PermissionCategory.for_tool("ReadFileTool") is PermissionCategory.READ
    assert PermissionCategory.for_tool("WriteFileTool") is PermissionCategory.WRITE
    assert PermissionCategory.for_tool("FetchWebTool") is PermissionCategory.WEB
    assert PermissionCategory.for_tool("BashCommandTool") is PermissionCategory.EXECUTE
    assert PermissionCategory.for_tool("EchoTool") is None


def test_get_allowed_tools_delegates_to_tool_manager() -> None:
    """get_allowed_tools returns whatever the tool manager builds from the grants."""

    class _RecordingToolManager:
        """Tool manager stub recording the permissions it was asked to build."""

        def __init__(self) -> None:
            self.received: dict[PermissionCategory, PermissionLevel] | None = None

        def create_tools(
            self, permissions: dict[PermissionCategory, PermissionLevel]
        ) -> list[str]:
            """Record the permissions and return a sentinel tool list.

            Args:
                permissions: the grants passed by the manager.

            Returns:
                A one-element sentinel list.
            """
            self.received = permissions
            return ["sentinel"]

    permissions = {PermissionCategory.READ: PermissionLevel.ASK}
    tool_manager = _RecordingToolManager()
    manager = PermissionManager(tool_manager, _FakeController(), permissions)

    assert manager.get_allowed_tools == ["sentinel"]
    assert tool_manager.received == permissions


def test_validate_auto_true_without_asking() -> None:
    """AUTO grants allow the call and never ask the controller."""
    manager = _manager({PermissionCategory.EXECUTE: PermissionLevel.AUTO})

    assert manager.validate(_call("BashCommandTool")) == (True, None)


def test_validate_ask_allows_on_allow_decision() -> None:
    """An ALLOW response with no message allows the call with no message."""
    controller = _FakeController(PermissionResponse(decision=Decision.ALLOW))
    manager = _manager({PermissionCategory.WRITE: PermissionLevel.ASK}, controller)

    assert manager.validate(_call("WriteFileTool")) == (True, None)
    assert isinstance(controller.requests[0], PermissionRequest)
    assert controller.requests[0].call.name == "WriteFileTool"


def test_validate_ask_allows_with_note() -> None:
    """An ALLOW response with a message wraps it in a report-then-answer note."""
    controller = _FakeController(
        PermissionResponse(decision=Decision.ALLOW, message="be careful")
    )
    manager = _manager({PermissionCategory.WRITE: PermissionLevel.ASK}, controller)

    allowed, message = manager.validate(_call("WriteFileTool"))

    assert allowed is True
    assert isinstance(message, UserMessage)
    assert message.display_text == "be careful"
    assert "be careful" in message.content


@pytest.mark.parametrize("note", [None, ""])
def test_validate_ask_allows_with_empty_note(note: str | None) -> None:
    """An ALLOW response with no/empty message allows with no message."""
    controller = _FakeController(
        PermissionResponse(decision=Decision.ALLOW, message=note)
    )
    manager = _manager({PermissionCategory.WRITE: PermissionLevel.ASK}, controller)

    assert manager.validate(_call("WriteFileTool")) == (True, None)


def test_validate_ask_denies_without_reason() -> None:
    """A DENY response with no message returns a bare denial tool error."""
    controller = _FakeController(PermissionResponse(decision=Decision.DENY))
    manager = _manager({PermissionCategory.WRITE: PermissionLevel.ASK}, controller)

    content = "Tool call WriteFileTool denied by user."
    assert manager.validate(_call("WriteFileTool")) == (
        False,
        ToolErrorMessage(content=content, id="call_1"),
    )


def test_validate_ask_denies_with_reason() -> None:
    """A DENY response with a message appends it to the denial as the reason."""
    controller = _FakeController(
        PermissionResponse(decision=Decision.DENY, message="use the read tool instead")
    )
    manager = _manager({PermissionCategory.WRITE: PermissionLevel.ASK}, controller)

    allowed, message = manager.validate(_call("WriteFileTool"))

    assert allowed is False
    assert isinstance(message, ToolErrorMessage)
    assert message.id == "call_1"
    assert "use the read tool instead" in message.content


def test_validate_absent_category_denies() -> None:
    """Calls to ungranted categories are denied without asking the controller."""
    manager = _manager({PermissionCategory.READ: PermissionLevel.ASK})

    content = "Tool WriteFileTool does not exist."
    assert manager.validate(_call("WriteFileTool")) == (
        False,
        ToolErrorMessage(content=content, id="call_1"),
    )


def test_validate_uncategorized_denies() -> None:
    """Calls to tools that map to no category are denied."""
    manager = _manager({PermissionCategory.READ: PermissionLevel.AUTO})

    content = "Tool EchoTool does not exist."
    assert manager.validate(_call("EchoTool")) == (
        False,
        ToolErrorMessage(content=content, id="call_1"),
    )


def test_add_and_remove() -> None:
    """Removing a grant revokes access and re-adding it restores access."""
    manager = _manager()

    manager.remove_permission(PermissionCategory.WEB)
    content = "Tool SearchWebTool does not exist."
    assert manager.validate(_call("SearchWebTool")) == (
        False,
        ToolErrorMessage(content=content, id="call_1"),
    )

    manager.add_permission(PermissionCategory.WEB, PermissionLevel.AUTO)
    assert manager.validate(_call("SearchWebTool")) == (True, None)


def test_add_permission_defaults_to_ask() -> None:
    """add_permission grants at the ASK level when no level is given."""
    manager = _manager()
    manager.remove_permission(PermissionCategory.READ)

    manager.add_permission(PermissionCategory.READ)

    assert manager.get_category_permission(PermissionCategory.READ) is (
        PermissionLevel.ASK
    )


def test_set_permission_level_changes_existing_grant() -> None:
    """set_permission_level updates the level of an existing grant in place."""
    manager = _manager({PermissionCategory.READ: PermissionLevel.ASK})

    manager.set_permission_level(PermissionCategory.READ, PermissionLevel.AUTO)

    assert (
        manager.get_category_permission(PermissionCategory.READ) is PermissionLevel.AUTO
    )


def test_set_permission_level_unknown_category_raises() -> None:
    """set_permission_level raises when the category has no grant."""
    manager = _manager({PermissionCategory.READ: PermissionLevel.ASK})

    with pytest.raises(ValueError):
        manager.set_permission_level(PermissionCategory.WEB, PermissionLevel.AUTO)


def test_get_category_permission_unknown_category_raises() -> None:
    """get_category_permission raises when the category has no grant."""
    manager = _manager({PermissionCategory.READ: PermissionLevel.ASK})

    with pytest.raises(ValueError):
        manager.get_category_permission(PermissionCategory.WEB)


def test_repr_lists_granted_categories_and_levels() -> None:
    """The repr maps each granted category to its level, in insertion order."""
    manager = _manager(
        {
            PermissionCategory.READ: PermissionLevel.ASK,
            PermissionCategory.WRITE: PermissionLevel.AUTO,
        }
    )

    assert repr(manager) == "PermissionManager(READ=ASK, WRITE=AUTO)"


def test_no_permissions_arg_grants_all_categories_at_ask() -> None:
    """Omitting permissions grants every category at ASK (the default manager)."""
    manager = _manager()

    for category in PermissionCategory:
        assert manager.get_category_permission(category) is PermissionLevel.ASK
