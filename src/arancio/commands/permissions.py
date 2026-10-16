"""Permissions command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.state_change import StateChangeCommand
from arancio.core.messages import ErrorMessage
from arancio.core.permissions.types import PermissionCategory, PermissionLevel

if TYPE_CHECKING:
    from arancio.sessions.manager import SessionManager
    from arancio.settings.manager import SettingsManager


class PermissionsCommand(StateChangeCommand):
    """Command that gets or sets one or all permission categories' autonomy levels."""

    name = "permissions"
    description = "Get or set a permission category's autonomy level, or all at once."

    @classmethod
    def execute(
        cls,
        category: str,
        settings_manager: SettingsManager,
        session_manager: SessionManager,
        level: str | None = None,
    ) -> str | ErrorMessage:
        """Report or replace the autonomy level for one or every permission category.

        With only ``category``, reports its currently set level, or that none
        is set when the category is at :attr:`PermissionLevel.NONE`. With
        ``level`` too, replaces it and applies/persists the change; the
        literal word ``"null"`` (case-insensitive) instead removes the grant,
        setting the category to :attr:`PermissionLevel.NONE` so its tools are
        never instantiated. The literal word ``"all"`` (case-insensitive) in
        place of ``category`` acts on every currently registered category at
        once instead of a single one. Both arguments are case-insensitive and
        must match an existing category (or ``"all"``) or level; anything
        else raises.

        Args:
            category: the permission category's name (e.g. ``"read"``), or
                ``"all"`` for every category; bound to the prompt's first
                word.
            settings_manager: the manager used to read, apply and persist the
                permission grants.
            session_manager: the active session manager, whose current
                session's configuration is updated to match and whose
                recorder persists a mutating change; unused when only
                reading the current level.
            level: the new autonomy level (e.g. ``"ask"``/``"auto"``), or
                ``"null"`` to remove the grant; bound to the prompt's second
                word. Omitted to only report the current level(s).

        Returns:
            The current level(s) (when only reading) or confirmation text
            naming the newly set level, or the persistence error notice when
            saving a mutating change failed.

        Raises:
            ValueError: when ``category`` does not name an existing category
                or ``"all"``, or ``level`` is given but does not name an
                existing level.
        """
        if category.lower() == "all":
            categories = tuple(PermissionCategory)
        else:
            try:
                categories = (PermissionCategory[category.upper()],)
            except KeyError:
                valid = ", ".join(
                    ["all"] + [member.name.lower() for member in PermissionCategory]
                )
                raise ValueError(
                    f"Unknown permission category: {category!r}. "
                    f"Valid categories: {valid}."
                ) from None

        if level is None:
            lines = []
            for resolved_category in categories:
                name = resolved_category.name.lower()
                current_level = settings_manager.settings.permissions[resolved_category]
                if current_level is PermissionLevel.NONE:
                    lines.append(f"No {name} permission set")
                else:
                    lines.append(f"{name} permission level: {current_level.value}")
            # two trailing spaces force a markdown hard line break, since the
            # result renders through a Markdown widget where a lone "\n" is
            # just a soft break (collapsed to a space)
            return "  \n".join(lines)

        if level.lower() == "null":
            new_level = PermissionLevel.NONE
        else:
            try:
                new_level = PermissionLevel(level.lower())
            except ValueError:
                other_levels = [
                    m.value for m in PermissionLevel if m is not PermissionLevel.NONE
                ]
                valid = ", ".join(["null"] + other_levels)
                raise ValueError(
                    f"Unknown permission level: {level!r}. Valid levels: {valid}."
                ) from None

        for resolved_category in categories:
            settings_manager.settings.permissions[resolved_category] = new_level
        settings_manager.apply()
        for resolved_category in categories:
            settings_manager.save_permission(resolved_category)
        cls._apply_configuration(session_manager, settings_manager)

        if len(categories) > 1:
            confirmation = (
                "all permissions removed"
                if new_level is PermissionLevel.NONE
                else f"all permission levels set to {new_level.value}"
            )
        else:
            name = categories[0].name.lower()
            confirmation = (
                f"{name} permission removed"
                if new_level is PermissionLevel.NONE
                else f"{name} permission level set to {new_level.value}"
            )
        return cls._persist_state_change(session_manager, confirmation)
