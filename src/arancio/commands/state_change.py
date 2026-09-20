"""Abstract interface for commands that change saved session state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import BaseCommand
from arancio.sessions.session import SessionConfiguration

if TYPE_CHECKING:
    from arancio.core.messages import ErrorMessage
    from arancio.sessions.manager import SessionManager
    from arancio.settings.manager import SettingsManager


class StateChangeCommand(BaseCommand):
    """Base class for commands that change the active session's saved state.

    A ``state_changed`` record snapshots the session's ``configuration``,
    ``working_directory`` and ``name`` together, and mutating those fields is
    the command's own job — :class:`arancio.sessions.recorder.SessionRecorder`
    only persists them. A subclass changes the field it owns, then calls
    :meth:`_persist_state_change` to write the snapshot and turn a failed
    write into the result the user sees. A subclass changing the session's
    configuration rather than its directory or name reaches
    :meth:`_apply_configuration` for the change itself.
    """

    @classmethod
    def _apply_configuration(
        cls, session_manager: SessionManager, settings_manager: SettingsManager
    ) -> None:
        """Snapshot the live settings onto the active session's configuration.

        Args:
            session_manager: the active session manager, whose current
                session's configuration is updated.
            settings_manager: the manager holding the just-applied settings.
        """
        session = session_manager.current
        session.configuration = SessionConfiguration.from_settings(
            settings_manager.settings
        )

    @classmethod
    def _persist_state_change(
        cls, session_manager: SessionManager, confirmation: str
    ) -> str | ErrorMessage:
        """Persist the session's just-mutated state, or report the failure instead.

        The caller is responsible for mutating the relevant field(s) on the
        session before calling this — this only persists and reports.

        Args:
            session_manager: the active session manager, whose recorder
                persists the current session's state.
            confirmation: the text to return when the write succeeds.

        Returns:
            ``confirmation``, or the persistence error notice when saving
            failed.
        """
        error = session_manager.session_recorder.state_changed()
        return error if error is not None else confirmation
