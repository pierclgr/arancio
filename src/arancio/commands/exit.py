"""Input adapter for the exit command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import Command

if TYPE_CHECKING:
    from arancio.application.base import Application


class ExitCommand(Command):
    """Quit the application."""

    name = "exit"
    description = "Quit the application."

    @classmethod
    def execute(cls, application: Application) -> None:
        """Delegate the parsed invocation to Application.

        Args:
            application: the operation facade supplied by the executor.
        """
        application.exit()
