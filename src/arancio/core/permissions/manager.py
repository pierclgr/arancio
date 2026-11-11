"""Permission grants and the resolution of each tool call against them."""

from collections.abc import Generator, Set

from arancio.core.events import (
    CoreAnswer,
    Decision,
    PermissionRequest,
    PermissionResponse,
)
from arancio.core.hooks.base import Hook
from arancio.core.hooks.manager import HookManager
from arancio.core.messages.base import Message
from arancio.core.messages.tool import ToolCallMessage
from arancio.core.permissions.constants import DEFAULT_PERMISSION_LEVELS
from arancio.core.permissions.types import (
    PermissionCategory,
    PermissionDecision,
    PermissionLevel,
    PermissionOutcome,
)


class PermissionManager:
    """Hold the agent's permission grants and resolve each tool call.

    The manager maps every :class:`PermissionCategory` to its
    :class:`PermissionLevel`. It builds no tools and contacts no frontend: an
    ``ask`` grant yields a :class:`~arancio.core.events.PermissionRequest`
    from :meth:`validate` and waits for the caller to send the answer.

    Attributes:
        _permissions: mapping of every category to its permission level.
        _hook_manager: the hook manager :meth:`validate` dispatches through.
    """

    def __init__(
        self,
        hook_manager: HookManager,
        permissions: dict[PermissionCategory, PermissionLevel] | None = None,
    ) -> None:
        """Initialize the manager with its hook manager and grants.

        Args:
            hook_manager: the hook manager :meth:`validate` dispatches
                through; this manager provides itself as
                ``permission_manager``.
            permissions: initial category-to-level grants, covering every
                category. When omitted (``None``), built-in categories use their
                configured defaults and other categories default to ASK.
        """
        if permissions is None:
            permissions = {
                category: DEFAULT_PERMISSION_LEVELS.get(category, PermissionLevel.ASK)
                for category in PermissionCategory
            }
        self._permissions: dict[PermissionCategory, PermissionLevel] = permissions
        self._hook_manager: HookManager = hook_manager
        self._hook_manager.provide(permission_manager=self)

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

    def add(
        self,
        permission_category: PermissionCategory,
        permission_level: PermissionLevel = PermissionLevel.ASK,
    ) -> None:
        """Grant a category at the given level.

        Args:
            permission_category: the category to grant.
            permission_level: the permission level. Defaults to
                :attr:`PermissionLevel.ASK`.

        Raises:
            ValueError: when the category is already granted.
        """
        # a category created after construction (e.g. by a custom plugin)
        # has no grant yet, so it counts as NONE
        granted = self._permissions.get(permission_category, PermissionLevel.NONE)
        if granted is not PermissionLevel.NONE:
            raise ValueError(
                f"Permission already granted for category {permission_category}."
            )
        self._permissions[permission_category] = permission_level

    def remove(self, permission_category: PermissionCategory) -> None:
        """Revoke the grant for the given category.

        Args:
            permission_category: the category whose grant is removed.

        Raises:
            ValueError: when the category has no grant.
        """
        # a category created after construction has no grant to revoke
        if self._permissions.get(permission_category) in (None, PermissionLevel.NONE):
            raise ValueError(
                f"No permission granted for category {permission_category}."
            )
        self._permissions[permission_category] = PermissionLevel.NONE

    def set_level(
        self, permission_category: PermissionCategory, permission_level: PermissionLevel
    ) -> None:
        """Change the permission level for the given category.

        Args:
            permission_category: the category whose level to update.
            permission_level: the new permission level.
        """
        self._permissions[permission_category] = permission_level

    def set(self, permissions: dict[PermissionCategory, PermissionLevel]) -> None:
        """Replace all grants with the given category-to-level mapping.

        Args:
            permissions: the new grants, covering every category, replacing
                the current ones wholesale.
        """
        self._permissions = permissions

    @property
    def allowed(self) -> Set[PermissionCategory]:
        """The categories whose tools may be built.

        Returns:
            Every category granted at a level other than
            :attr:`PermissionLevel.NONE`.
        """
        return {
            category
            for category, level in self._permissions.items()
            if level is not PermissionLevel.NONE
        }

    def validate(
        self, tool_call: ToolCallMessage, permission_category: PermissionCategory | None
    ) -> Generator[
        PermissionRequest, CoreAnswer, tuple[PermissionDecision, list[Message]]
    ]:
        """Decide whether a requested tool call may execute.

        Dispatches ``before_permission_check`` before resolving the call and
        ``after_permission_check`` once a decision exists, exactly once each
        regardless of which of :meth:`_resolve`'s outcomes is reached. Use it
        with ``yield from``: an ``ask`` grant yields a
        :class:`~arancio.core.events.PermissionRequest` and resumes with the
        caller's :class:`~arancio.core.events.PermissionResponse`. Closing the
        generator while it waits runs no after hook and decides nothing.

        Args:
            tool_call: the tool call the agent wants to execute.
            permission_category: the call's category from the tool catalog, or ``None``
                when the tool is unknown.

        Returns:
            The :class:`~arancio.core.permissions.types.PermissionDecision`
            resolving the call, and hook messages in dispatch order. The
            caller runs the tool and turns the decision into the messages the
            model sees.

        Yields:
            The permission request of an ``ask`` grant.
        """
        messages = self._hook_manager.run(
            Hook.BEFORE_PERMISSION_CHECK, tool_call=tool_call
        )
        permission_decision = yield from self._resolve(tool_call, permission_category)
        messages.extend(
            self._hook_manager.run(
                Hook.AFTER_PERMISSION_CHECK,
                tool_call=tool_call,
                permission_decision=permission_decision,
            )
        )
        return permission_decision, messages

    def _resolve(
        self, tool_call: ToolCallMessage, permission_category: PermissionCategory | None
    ) -> Generator[PermissionRequest, CoreAnswer, PermissionDecision]:
        """Resolve a tool call to a permission decision, without dispatching hooks.

        Kept separate from :meth:`validate` so hook dispatch there wraps
        exactly one before/after pair instead of tripling
        ``after_permission_check`` across the three branches below.

        Args:
            tool_call: the tool call the agent wants to execute.
            permission_category: the call's category, or ``None`` for an unknown tool.

        Returns:
            The resolved :class:`~arancio.core.permissions.types.PermissionDecision`.

        Yields:
            The permission request of an ``ask`` grant.

        Raises:
            TypeError: when the answer to a permission request is not a
                :class:`~arancio.core.events.PermissionResponse`.
            ValueError: when the answer does not match the pending request.
        """
        level = (
            self._permissions.get(permission_category, PermissionLevel.NONE)
            if permission_category is not None
            else PermissionLevel.NONE
        )

        # unavailable case: an unknown tool or a category that is not granted
        if level is PermissionLevel.NONE:
            return PermissionDecision(outcome=PermissionOutcome.UNAVAILABLE)

        # automatic case
        if level is PermissionLevel.AUTO:
            return PermissionDecision(outcome=PermissionOutcome.ALLOWED)

        # ask case: wait for the caller's answer; a wrong or missing answer
        # never approves the call
        permission_request = PermissionRequest(tool_call=tool_call)
        response = yield permission_request
        if not isinstance(response, PermissionResponse):
            raise TypeError(
                f"Permission request {permission_request.id} needs a "
                f"PermissionResponse, got {response!r}."
            )
        if response.request_id != permission_request.id:
            raise ValueError(
                f"Permission response for request {response.request_id!r} does "
                f"not match the pending request {permission_request.id!r}."
            )
        # a deny keeps the user's reason on the decision, the caller words the
        # refusal
        outcome = (
            PermissionOutcome.ALLOWED
            if response.decision is Decision.ALLOW
            else PermissionOutcome.DENIED
        )
        return PermissionDecision(outcome=outcome, note=response.message or None)
