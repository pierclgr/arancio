"""Input adapter for the reasoning-effort command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import Command

if TYPE_CHECKING:
    from arancio.application.base import Application
    from arancio.core.messages.base import ErrorMessage


class EffortCommand(Command):
    """Set the model's reasoning effort."""

    name = "effort"
    description = "Set the model's reasoning effort."

    @classmethod
    def execute(cls, level: str, application: Application) -> str | ErrorMessage | None:
        """Translate the command's null token and update reasoning effort.

        Args:
            level: the effort text, or null to disable reasoning.
            application: the operation facade supplied by the executor.

        Returns:
            The existing confirmation text or a persistence notice.
        """
        effort = None if level.lower() == "null" else level
        return application.update_settings(model={"reasoning": {"effort": effort}})
