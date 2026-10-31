"""Input adapter for the resume command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import Command

if TYPE_CHECKING:
    from arancio.application.base import Application
    from arancio.core.messages.base import Message


class ResumeCommand(Command):
    """Resume a saved session by identity or name."""

    name = "resume"
    description = "Resume a previous session by ID or name."
    joins_arguments = True

    @classmethod
    def execute(cls, query: str, application: Application) -> str | Message | None:
        """Delegate the parsed invocation to Application.

        Args:
            query: the parsed command argument.
            application: the operation facade supplied by the executor.

        Returns:
            The Application operation's confirmation, state or notices.
        """
        return application.resume(query)
