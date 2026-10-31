"""Input adapter for the clear command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import Command

if TYPE_CHECKING:
    from arancio.application.base import Application


class ClearCommand(Command):
    """Clear the chat history, starting a new conversation."""

    name = "clear"
    description = "Clear the chat history, starting a new conversation."

    @classmethod
    def execute(cls, application: Application) -> None:
        """Delegate the parsed invocation to Application.

        Args:
            application: the operation facade supplied by the executor.
        """
        application.new_session()
