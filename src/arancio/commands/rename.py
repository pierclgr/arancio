"""Rename command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.state_change import StateChangeCommand
from arancio.core.messages import ErrorMessage

if TYPE_CHECKING:
    from arancio.sessions.manager import SessionManager


class RenameCommand(StateChangeCommand):
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
            Confirmation text naming the session's new name, or the
            persistence error notice when saving the change failed.
        """
        session = session_manager.current
        session.name = new_name
        return cls._persist_state_change(
            session_manager, f"Session renamed to {new_name!r}"
        )
