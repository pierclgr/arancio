"""Input adapter for the model command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import Command

if TYPE_CHECKING:
    from arancio.application.base import Application
    from arancio.core.messages.base import ErrorMessage


class ModelCommand(Command):
    """Set the model name, keeping the current provider."""

    name = "model"
    description = "Set the model name, keeping the current provider."

    @classmethod
    def execute(cls, model_name: str, application: Application) -> str | ErrorMessage:
        """Delegate the parsed invocation to Application.

        Args:
            model_name: the parsed command argument.
            application: the operation facade supplied by the executor.

        Returns:
            The Application operation's confirmation, state or notices.
        """
        return application.update_settings(model_name=model_name)
