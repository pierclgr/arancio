"""Input adapter for the cd command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import Command

if TYPE_CHECKING:
    from arancio.application.base import Application
    from arancio.core.messages.base import ErrorMessage


class CdCommand(Command):
    """Change the working directory."""

    name = "cd"
    description = "Change the working directory."

    @classmethod
    def execute(cls, path: str, application: Application) -> str | ErrorMessage:
        """Delegate the parsed invocation to Application.

        Args:
            path: the parsed command argument.
            application: the operation facade supplied by the executor.

        Returns:
            The Application operation's confirmation, state or notices.
        """
        return application.change_directory(path)
