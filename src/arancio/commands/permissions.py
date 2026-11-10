"""Input adapter for reading or updating typed permission grants."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import Command
from arancio.core.permissions.types import PermissionLevel

if TYPE_CHECKING:
    from arancio.application.base import Application
    from arancio.core.messages.base import ErrorMessage


class PermissionsCommand(Command):
    """Get or set one or all permission categories' autonomy levels."""

    name = "permissions"
    description = "Get or set a permission category's autonomy level, or all at once."

    @classmethod
    def execute(
        cls,
        category: str,
        application: Application,
        level: str | None = None,
    ) -> str | ErrorMessage | None:
        """Resolve input tokens using Application's copied permission metadata.

        Args:
            category: a known category name or all, case-insensitively.
            application: the operation facade supplied by the executor.
            level: ask, auto or none; omission reads the selected grants.

        Returns:
            Existing level text, update confirmation, or a persistence notice.

        Raises:
            ValueError: when the supplied category or level is unknown.
        """
        levels = application.permission_levels()
        if category.lower() == "all":
            categories = tuple(levels)
        else:
            by_name = {member.name.lower(): member for member in levels}
            resolved = by_name.get(category.lower())
            if resolved is None:
                valid = ", ".join(["all"] + list(by_name))
                raise ValueError(
                    f"Unknown permission category: {category!r}. "
                    f"Valid categories: {valid}."
                )
            categories = (resolved,)

        if level is None:
            lines: list[str] = []
            for member in categories:
                name = member.name.lower()
                current = levels[member]
                lines.append(
                    f"No {name} permission set"
                    if current is PermissionLevel.NONE
                    else f"{name} permission level: {current.value}"
                )
            return "  \n".join(lines)

        if level.lower() == "none":
            new_level = PermissionLevel.NONE
        else:
            try:
                new_level = PermissionLevel(level.lower())
            except ValueError:
                other_levels = [
                    member.value
                    for member in PermissionLevel
                    if member is not PermissionLevel.NONE
                ]
                valid = ", ".join(["none"] + other_levels)
                raise ValueError(
                    f"Unknown permission level: {level!r}. Valid levels: {valid}."
                ) from None
        return application.update_settings(
            permissions={member: new_level for member in categories}
        )
