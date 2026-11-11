"""Input adapter for the fork command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import Command

if TYPE_CHECKING:
    from arancio.application.base import Application
    from arancio.core.messages.assistant import AssistantMessage
    from arancio.core.messages.base import ErrorMessage
    from arancio.sessions.state import SessionState


class ForkCommand(Command):
    """Fork the active session into a new one."""

    name = "fork"
    description = "Fork the active session into a new one."

    @classmethod
    def execute(
        cls, application: Application
    ) -> tuple[SessionState, AssistantMessage, list[ErrorMessage]]:
        """Delegate the parsed invocation to Application.

        Args:
            application: the operation facade supplied by the executor.

        Returns:
            The Application operation's confirmation, state or notices.
        """
        return application.fork()
