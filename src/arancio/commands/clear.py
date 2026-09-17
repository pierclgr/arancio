"""Clear command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import BaseCommand
from arancio.sessions.session import SessionConfiguration

if TYPE_CHECKING:
    from arancio.core.agents import Agent
    from arancio.sessions.manager import SessionManager
    from arancio.settings.manager import SettingsManager
    from arancio.ui.app import App


class ClearCommand(BaseCommand):
    """Command that clears the chat history, starting a new conversation."""

    name = "clear"
    description = "Clear the chat history, starting a new conversation."

    @classmethod
    def execute(
        cls,
        agent: Agent,
        application: App,
        settings_manager: SettingsManager,
        session_manager: SessionManager,
    ) -> None:
        """Empty the agent's conversation history and the displayed log.

        Args:
            agent: the agent whose conversation history is emptied.
            application: the running app whose log is cleared to match.
            settings_manager: the active settings manager.
            session_manager: the active session manager.
        """
        settings_manager.reset_to_global()
        session_manager.discard_and_create(
            working_directory=application.working_directory,
            configuration=SessionConfiguration.from_settings(settings_manager.settings),
        )
        agent.clear_history()
        application.clear_log()
