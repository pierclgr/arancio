"""Permissions command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import BaseCommand
from arancio.core.permissions.types import PermissionCategory, PermissionLevel

if TYPE_CHECKING:
    from arancio.settings.manager import SettingsManager


class PermissionsCommand(BaseCommand):
    """Command that gets or sets a permission category's autonomy level."""

    name = "permissions"
    description = "Get or set a permission category's autonomy level."

    @classmethod
    def execute(
        cls,
        category: str,
        settings_manager: SettingsManager,
        level: str | None = None,
    ) -> str:
        """Report or replace the autonomy level for a permission category.

        With only ``category``, reports its currently set level, or that none
        is set when the category is at :attr:`PermissionLevel.NONE`. With
        ``level`` too, replaces it and applies/persists the change; the
        literal word ``"null"`` (case-insensitive) instead removes the grant,
        setting the category to :attr:`PermissionLevel.NONE` so its tools are
        never instantiated. Both arguments are case-insensitive and must
        match an existing category or level; anything else raises.

        Args:
            category: the permission category's name (e.g. ``"read"``),
                bound to the prompt's first word.
            settings_manager: the manager used to read, apply and persist the
                permission grants.
            level: the new autonomy level (e.g. ``"ask"``/``"auto"``), or
                ``"null"`` to remove the grant; bound to the prompt's second
                word. Omitted to only report the current level.

        Returns:
            The current level (when only reading) or confirmation text naming
            the newly set level.

        Raises:
            ValueError: when ``category`` does not name an existing category,
                or ``level`` is given but does not name an existing level.
        """
        try:
            resolved_category = PermissionCategory[category.upper()]
        except KeyError:
            valid = ", ".join(member.name.lower() for member in PermissionCategory)
            raise ValueError(
                f"Unknown permission category: {category!r}. Valid categories: {valid}."
            ) from None

        if level is None:
            current_level = settings_manager.settings.permissions[resolved_category]
            if current_level is PermissionLevel.NONE:
                return f"No {resolved_category.name.lower()} permission set"
            return (
                f"{resolved_category.name.lower()} permission level: "
                f"{current_level.value}"
            )

        if level.lower() == "null":
            settings_manager.settings.permissions[resolved_category] = (
                PermissionLevel.NONE
            )
            settings_manager.apply()
            settings_manager.save()
            return f"{resolved_category.name.lower()} permission removed"

        try:
            new_level = PermissionLevel(level.lower())
        except ValueError:
            valid = ", ".join(
                ["null"]
                + [m.value for m in PermissionLevel if m is not PermissionLevel.NONE]
            )
            raise ValueError(
                f"Unknown permission level: {level!r}. Valid levels: {valid}."
            ) from None

        settings_manager.settings.permissions[resolved_category] = new_level
        settings_manager.apply()
        settings_manager.save()
        return (
            f"{resolved_category.name.lower()} permission level set to "
            f"{new_level.value}"
        )
