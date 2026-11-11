"""Input adapter for the provider command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import Command

if TYPE_CHECKING:
    from arancio.application.base import Application
    from arancio.core.messages.base import ErrorMessage


class ProviderCommand(Command):
    """Set the provider, keeping the current model name."""

    name = "provider"
    description = "Set the provider, keeping the current model name."

    @classmethod
    def execute(cls, provider: str, application: Application) -> str | ErrorMessage:
        """Delegate the parsed invocation to Application.

        Args:
            provider: the parsed command argument.
            application: the operation facade supplied by the executor.

        Returns:
            The Application operation's confirmation, state or notices.
        """
        return application.update_settings(model={"provider": provider})
