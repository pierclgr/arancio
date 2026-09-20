"""Rename command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import BaseCommand
from arancio.core.messages import ErrorMessage

if TYPE_CHECKING:
    from arancio.sessions.manager import SessionManager


class RenameCommand(BaseCommand):
    """Command that renames the active chat session."""

    name = "rename"
    description = "Rename the active session."

    @classmethod
    def execute(
        cls, new_name: str, session_manager: SessionManager
    ) -> str | ErrorMessage:
        """Set the active session's display name and persist it.

        ``state_changed`` always records a complete snapshot, so configuration
        and working directory are carried over unchanged along with the new
        name.

        Args:
            new_name: the session's new name, bound to the prompt's first
                word.
            session_manager: the active session manager, whose current
                session is renamed.

        Returns:
            Confirmation text naming the session and its new name, or the
            persistence error notice when saving the change failed.
        """
        session = session_manager.get_current_session()
        session.name = new_name
        error = session_manager.session_recorder.state_changed()
        if error is not None:
            return error
        return f"Session {session.id} renamed to {new_name!r}"
