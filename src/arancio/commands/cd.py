"""Change working directory command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import BaseCommand

if TYPE_CHECKING:
    from arancio.ui.app import App


class CdCommand(BaseCommand):
    """Command that moves the working directory."""

    name = "cd"
    description = "Change the working directory."

    @classmethod
    def execute(cls, path: str, application: App) -> str:
        """Move the working directory to ``path``.

        Resolution and validation are left to
        :meth:`~arancio.ui.app.App.set_working_directory`, which raises
        ``ValueError`` when the target does not exist or is not a directory;
        the executor turns that into an error message rather than crashing.

        Args:
            path: the target directory, bound to the prompt's first word. A
                relative path is resolved against the working directory
                currently set; an absolute path is taken as-is. ``~`` is
                expanded.
            application: the running app whose working directory is moved.

        Returns:
            Confirmation text naming the new working directory.
        """
        application.set_working_directory(path)
        return f"Working directory set to {application.working_directory}"
