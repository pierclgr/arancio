"""Central permission registry filtering and gating the agent's tool use."""

from arancio.core.controllers.base import Controller
from arancio.core.controllers.requests import PermissionRequest
from arancio.core.controllers.responses import Decision
from arancio.core.hooks.manager import HookManager
from arancio.core.hooks.types import Hook
from arancio.core.messages import ToolCallMessage
from arancio.core.permissions.types import (
    PermissionCategory,
    PermissionDecision,
    PermissionLevel,
    PermissionOutcome,
)
from arancio.core.tools.base import BaseTool
from arancio.core.tools.manager import ToolManager


class PermissionManager:
    """Hold the agent's permissions, create its tools and gate tool calls.

    The manager maps every :class:`PermissionCategory` to its
    :class:`PermissionLevel`, and owns a :class:`ToolManager` that builds the
    tools for the granted categories. Tools whose category is at
    :attr:`PermissionLevel.NONE` are never created; calls to permitted tools
    are gated by their level.

    Attributes:
        _permissions: mapping of every category to its permission level.
        _tool_manager: the tool manager that creates tools from the grants.
        _controller: the controller used to ask the user about ``ask`` grants.
        _hook_manager: the hook manager :meth:`validate` dispatches through.
    """

    def __init__(
        self,
        tool_manager: ToolManager,
        controller: Controller,
        hook_manager: HookManager,
        permissions: dict[PermissionCategory, PermissionLevel] | None = None,
    ) -> None:
        """Initialize the manager with a tool manager, controller and grants.

        Args:
            tool_manager: the tool manager used to create the agent's tools and
                to resolve which tools are available.
            controller: the controller through which the manager asks the user
                to approve or deny ``ask`` grants.
            hook_manager: the hook manager :meth:`validate` dispatches
                through.
            permissions: initial category-to-level grants, covering every
                category. When omitted (``None``) every category is granted at
                :attr:`PermissionLevel.ASK`.
        """
        if permissions is None:
            permissions = {
                category: PermissionLevel.ASK for category in PermissionCategory
            }
        self._permissions: dict[PermissionCategory, PermissionLevel] = permissions
        self._tool_manager: ToolManager = tool_manager
        self._controller: Controller = controller
        self._hook_manager: HookManager = hook_manager

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the permission grants.

        Returns:
            A compact string mapping each category to its level.
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
        if self._permissions[category] is not PermissionLevel.NONE:
            raise ValueError(f"Permission already granted for category {category}.")
        self._permissions[category] = level

    def remove_permission(self, category: PermissionCategory) -> None:
        """Revoke the grant for the given category.

        Args:
            category: the category whose grant is removed.

        Raises:
            ValueError: when the category has no grant.
        """
        if self._permissions[category] is PermissionLevel.NONE:
            raise ValueError(f"No permission granted for category {category}.")
        self._permissions[category] = PermissionLevel.NONE

    def set_permission_level(
        self, category: PermissionCategory, level: PermissionLevel
    ) -> None:
        """Change the permission level for the given category.

        Args:
            category: the category whose level to update.
            level: the new permission level.
        """
        self._permissions[category] = level

    def set_permissions(
        self, permissions: dict[PermissionCategory, PermissionLevel]
    ) -> None:
        """Replace all grants with the given category-to-level mapping.

        Args:
            permissions: the new grants, covering every category, replacing
                the current ones wholesale.
        """
        self._permissions = permissions

    @property
    def allowed_tools(self) -> list[BaseTool]:
        """Create the tools the agent may access from the current grants.

        Returns:
            Tool instances for the granted categories, built by the tool
            manager.
        """
        return self._tool_manager.create_tools(self._permissions)

    def validate(self, call: ToolCallMessage) -> PermissionDecision:
        """Decide whether a requested tool call may execute.

        Dispatches ``before_permission_check`` before resolving the call and
        ``after_permission_check`` once resolved, exactly once each
        regardless of which of :meth:`_resolve`'s outcomes is reached.
        Allows ``auto`` grants without asking and asks the user through the
        controller for ``ask`` grants. A call whose category is at
        :attr:`PermissionLevel.NONE` (or maps to no category) resolves as
        unavailable. The
        controller's :class:`~arancio.core.controllers.responses.PermissionResponse`
        either allows the call (optionally with a note for the model) or denies
        it (optionally with a reason for the model).

        Args:
            call: the tool call the agent wants to execute.

        Returns:
            The :class:`~arancio.core.permissions.types.PermissionDecision`
            resolving the call. The caller runs the tool and turns the decision
            into the messages the model sees.
        """
        self._hook_manager.run(Hook.BEFORE_PERMISSION_CHECK, call=call)
        decision = self._resolve(call)
        self._hook_manager.run(
            Hook.AFTER_PERMISSION_CHECK, call=call, decision=decision
        )
        return decision

    def _resolve(self, call: ToolCallMessage) -> PermissionDecision:
        """Resolve a tool call to a permission decision, without dispatching hooks.

        Kept separate from :meth:`validate` so hook dispatch there wraps
        exactly one before/after pair instead of tripling
        ``after_permission_check`` across the three branches below.

        Args:
            call: the tool call the agent wants to execute.

        Returns:
            The resolved :class:`~arancio.core.permissions.types.PermissionDecision`.
        """
        # unavailable case
        if not self._tool_manager.is_tool_available(call.name, self._permissions):
            return PermissionDecision(outcome=PermissionOutcome.UNAVAILABLE)

        category = PermissionCategory.for_tool(call.name)

        # automatic case
        if self._permissions[category] is PermissionLevel.AUTO:
            return PermissionDecision(outcome=PermissionOutcome.ALLOWED)

        # ask case: delegate to the controller and map its response; a deny
        # keeps the user's reason on the decision, the caller words the refusal
        response = self._controller.request(PermissionRequest(call))
        outcome = (
            PermissionOutcome.ALLOWED
            if response.decision is Decision.ALLOW
            else PermissionOutcome.DENIED
        )
        return PermissionDecision(outcome=outcome, note=response.message or None)
