"""Input adapter for the rename command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import Command

if TYPE_CHECKING:
    from arancio.application.base import Application
    from arancio.core.messages.base import ErrorMessage


class RenameCommand(Command):
    """Rename the active session."""

    name = "rename"
    description = "Rename the active session."
    joins_arguments = True

    @classmethod
    def execute(cls, new_name: str, application: Application) -> str | ErrorMessage:
        """Delegate the parsed invocation to Application.

        Args:
            new_name: the parsed command argument.
            application: the operation facade supplied by the executor.

        Returns:
            The Application operation's confirmation, state or notices.
        """
        return application.rename(new_name)
