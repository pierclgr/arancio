"""Clear command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import BaseCommand

if TYPE_CHECKING:
    from arancio.core.agents import Agent
    from arancio.ui.app import App


class ClearCommand(BaseCommand):
    """Command that clears the chat history, starting a new conversation."""

    name = "clear"
    description = "Clear the chat history, starting a new conversation."

    @classmethod
    def execute(cls, agent: Agent, application: App) -> None:
        """Empty the agent's conversation history and the displayed log.

        Args:
            agent: the agent whose conversation history is emptied.
            application: the running app whose log is cleared to match.
        """
        agent.clear_history()
        application.clear_log()
