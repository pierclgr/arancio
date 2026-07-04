"""Exit command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import BaseCommand

if TYPE_CHECKING:
    from arancio.ui.app import App


class ExitCommand(BaseCommand):
    """Command that quits the application."""

    name = "exit"
    description = "Quit the application."

    @classmethod
    def execute(cls, application: App, **kwargs) -> None:
        """Quit the application.

        Args:
            application: the running app to quit.
            **kwargs: absorbs any argument this command does not need.
        """
        application.exit()
