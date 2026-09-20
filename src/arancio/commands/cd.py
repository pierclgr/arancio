"""Change working directory command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.state_change import StateChangeCommand
from arancio.core.messages import ErrorMessage

if TYPE_CHECKING:
    from arancio.sessions.manager import SessionManager
    from arancio.ui.app import App


class CdCommand(StateChangeCommand):
    """Command that moves the working directory."""

    name = "cd"
    description = "Change the working directory."

    @classmethod
    def execute(
        cls, path: str, application: App, session_manager: SessionManager
    ) -> str | ErrorMessage:
        """Move the working directory to ``path`` and record it on the session.

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
            session_manager: the active session manager, whose current
                session's working directory is updated to match and whose
                recorder persists the change.

        Returns:
            Confirmation text naming the new working directory, or the
            persistence error notice when saving the change failed.
        """
        application.set_working_directory(path)
        session = session_manager.get_current_session()
        session.working_directory = application.working_directory
        return cls._persist_state_change(
            session_manager, f"Working directory set to {application.working_directory}"
        )
