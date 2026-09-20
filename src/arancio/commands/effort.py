"""Effort command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.state_change import StateChangeCommand
from arancio.core.messages import ErrorMessage

if TYPE_CHECKING:
    from arancio.sessions.manager import SessionManager
    from arancio.settings.manager import SettingsManager
    from arancio.ui.app import App


class EffortCommand(StateChangeCommand):
    """Command that sets the model's thinking effort."""

    name = "effort"
    description = "Set the model's thinking effort."

    @classmethod
    def execute(
        cls,
        level: str,
        application: App,
        settings_manager: SettingsManager,
        session_manager: SessionManager,
    ) -> str | ErrorMessage:
        """Replace the thinking effort and apply it.

        Any text is accepted as the effort level; no fixed set of values is
        enforced. The literal word ``"null"`` (case-insensitive) disables
        thinking entirely. Accessing ``settings.model_id`` below raises
        ``ValueError`` before anything is persisted when no model is
        currently configured.

        Args:
            level: the new thinking effort, bound to the prompt's first word;
                ``"null"`` (any case) disables thinking.
            application: the running app whose toolbar is refreshed with the
                new effort level.
            settings_manager: the manager used to apply and persist the
                change.
            session_manager: the active session manager, whose current
                session's configuration is updated to match and whose
                recorder persists the change.

        Returns:
            Confirmation text naming the new effort level, or the
            persistence error notice when saving the change failed.
        """
        settings_manager.settings.model_id
        new_effort = None if level.lower() == "null" else level
        settings_manager.settings.thinking_effort = new_effort
        settings_manager.apply()
        settings_manager.save_thinking_effort()
        application.set_displayed_effort(new_effort)
        cls._apply_configuration(session_manager, settings_manager)
        shown_effort = new_effort if new_effort is not None else "null"
        return cls._persist_state_change(
            session_manager, f"Thinking effort set to {shown_effort}"
        )
