"""Central permission registry filtering and gating the agent's tool use."""

from arancio.core.controllers.base import Controller
from arancio.core.controllers.requests import PermissionRequest
from arancio.core.controllers.responses import Decision
from arancio.core.tools.base import BaseTool
from arancio.core.tools.manager import ToolManager
from arancio.core.types.messages import (
    Message,
    ToolCallMessage,
    ToolErrorMessage,
    UserMessage,
)
from arancio.core.types.permissions import PermissionCategory, PermissionLevel


class PermissionManager:
    """Hold the agent's permissions, create its tools and gate tool calls.

    The manager maps each granted :class:`PermissionCategory` to its
    :class:`PermissionLevel`, and owns a :class:`ToolManager` that builds the
    tools for the granted categories. Tools whose category has no grant are
    never created; calls to permitted tools are gated by their level.

    Attributes:
        _permissions: mapping of granted category to its permission level.
        _tool_manager: the tool manager that creates tools from the grants.
        _controller: the controller used to ask the user about ``ask`` grants.
    """

    def __init__(
        self,
        tool_manager: ToolManager,
        controller: Controller,
        permissions: dict[PermissionCategory, PermissionLevel] | None = None,
    ) -> None:
        """Initialize the manager with a tool manager, controller and grants.

        Args:
            tool_manager: the tool manager used to create the agent's tools and
                to resolve which tools are available.
            controller: the controller through which the manager asks the user
                to approve or deny ``ask`` grants.
            permissions: initial category-to-level grants. When omitted
                (``None``) every category is granted at
                :attr:`PermissionLevel.ASK`; pass an explicit ``{}`` to start
                with no grants.
        """
        if not permissions:
            permissions = {
                category: PermissionLevel.ASK for category in PermissionCategory
            }
        self._permissions: dict[PermissionCategory, PermissionLevel] = permissions
        self._tool_manager: ToolManager = tool_manager
        self._controller: Controller = controller

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the permission grants.

        Returns:
            A compact string mapping each granted category to its level.
        """
        grants = ", ".join(
            f"{category.name}={level.name}"
            for category, level in self._permissions.items()
        )
        return f"{type(self).__name__}({grants})"

    def add_permission(
        self,
        category: PermissionCategory,
        level: PermissionLevel = PermissionLevel.ASK,
    ) -> None:
        """Grant a category at the given level.

        Args:
            category: the category to grant.
            level: the permission level. Defaults to
                :attr:`PermissionLevel.ASK`.

        Raises:
            ValueError: when the category is already granted.
        """
        if category in self._permissions:
            raise ValueError(f"Permission already granted for category {category}.")
        self._permissions[category] = level

    def remove_permission(self, category: PermissionCategory) -> None:
        """Revoke the grant for the given category.

        Args:
            category: the category whose grant is removed.

        Raises:
            ValueError: when the category has no grant.
        """
        if category not in self._permissions:
            raise ValueError(f"No permission granted for category {category}.")
        del self._permissions[category]

    def get_category_permission(self, category: PermissionCategory) -> PermissionLevel:
        """Return the permission level granted for the given category.

        Args:
            category: the category to look up.

        Returns:
            The granted level.

        Raises:
            ValueError: when the category has no grant.
        """
        if category not in self._permissions:
            raise ValueError(f"No permission granted for category {category}.")
        return self._permissions[category]

    def set_permission_level(
        self, category: PermissionCategory, level: PermissionLevel
    ) -> None:
        """Change the permission level of the grant for the given category.

        Args:
            category: the category whose grant to update.
            level: the new permission level.

        Raises:
            ValueError: when the category has no grant to update.
        """
        if category not in self._permissions:
            raise ValueError(f"No permission granted for category {category}.")
        self._permissions[category] = level

    def set_permissions(
        self, permissions: dict[PermissionCategory, PermissionLevel]
    ) -> None:
        """Replace all grants with the given category-to-level mapping.

        Args:
            permissions: the new grants, replacing the current ones wholesale.
        """
        self._permissions = permissions

    @property
    def get_allowed_tools(self) -> list[BaseTool]:
        """Create the tools the agent may access from the current grants.

        Returns:
            Tool instances for the granted categories, built by the tool
            manager.
        """
        return self._tool_manager.create_tools(self._permissions)

    def validate(self, call: ToolCallMessage) -> tuple[bool, Message | None]:
        """Decide whether a requested tool call may execute.

        Allows ``auto`` grants without asking and asks the user through the
        controller for ``ask`` grants. A call whose category has no grant (or
        maps to no category) is denied with a not-permitted error. The
        controller's :class:`~arancio.core.controllers.responses.PermissionResponse`
        either allows the call (optionally with a note for the model) or denies
        it (optionally with a reason for the model).

        Args:
            call: the tool call the agent wants to execute.

        Returns:
            An ``(allowed, message)`` pair. ``message`` is a
            :class:`~arancio.core.types.messages.UserMessage` wrapping the note in a
            report-then-answer instruction when the call is allowed with one, a
            :class:`~arancio.core.types.messages.ToolErrorMessage` describing the denial
            or not-permitted reason when the call is refused, and ``None`` for a
            plain allow.
        """
        if not self._tool_manager.is_tool_available(call.name, self._permissions):
            message = f"Tool {call.name} does not exist."
            return False, ToolErrorMessage(
                content=message,
                id=call.id,
            )

        category = PermissionCategory.for_tool(call.name)

        # automatic case
        if self._permissions[category] is PermissionLevel.AUTO:
            return True, None

        # ask case: delegate to the controller and map its response
        response = self._controller.request(PermissionRequest(call))
        if response.decision is Decision.ALLOW:
            if response.message:
                instruction = (
                    f"While running tool {call.id}: {call.name}, user also "
                    f"noted: {response.message}. First report tool calling result, "
                    f"then answer user note."
                )
                return True, UserMessage(
                    content=instruction, display_text=response.message
                )
            return True, None

        # denied case
        content = f"Tool call {call.name} denied by user."
        if response.message:
            content += f" Additional information from user: {response.message}"

        return False, ToolErrorMessage(content=content, id=call.id)
