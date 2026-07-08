"""Provider command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import BaseCommand

if TYPE_CHECKING:
    from arancio.settings.manager import SettingsManager
    from arancio.ui.app import App


class ProviderCommand(BaseCommand):
    """Command that sets the provider, keeping the current model name."""

    name = "provider"
    description = "Set the provider, keeping the current model name."

    @classmethod
    def execute(
        cls, provider: str, application: App, settings_manager: SettingsManager
    ) -> str:
        """Replace the provider and apply it.

        Unlike :class:`~arancio.commands.model.ModelCommand`, this does not
        require a model name to already be configured — setting the provider
        alone is the first configuration step on a fresh install.

        Args:
            provider: the new provider prefix (e.g. ``"openai"``), bound to
                the prompt's first word; any text is accepted. The current
                model name is kept unchanged.
            application: the running app whose toolbar is refreshed with the
                new model id, when a model name is already configured too.
            settings_manager: the manager used to apply and persist the
                change.

        Returns:
            Confirmation text naming the new provider.
        """
        settings = settings_manager.settings
        settings.provider = provider
        settings_manager.apply()
        settings_manager.save()
        if settings.model_name:
            application.set_displayed_model_id(settings.model_id)
        return f"Provider set to {provider}"
