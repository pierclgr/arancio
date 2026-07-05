"""Model command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.base import BaseCommand

if TYPE_CHECKING:
    from arancio.settings.manager import SettingsManager
    from arancio.ui.app import App


class ModelCommand(BaseCommand):
    """Command that sets the model name, keeping the current provider."""

    name = "model"
    description = "Set the model name, keeping the current provider."

    @classmethod
    def execute(
        cls, model_name: str, application: App, settings_manager: SettingsManager
    ) -> str:
        """Replace the model name in the current model id and apply it.

        ``provider`` and ``model_name`` are stored as separate settings
        fields; the settings manager's ``apply`` joins them into the client's
        ``provider/model_name`` model id.

        Args:
            model_name: the new model name, bound to the prompt's first word;
                any text is accepted. The current model id's provider prefix
                (the text before its first ``/``) is kept unchanged.
            application: the running app whose toolbar is refreshed with the
                new model id.
            settings_manager: the manager used to apply and persist the
                change.

        Returns:
            Confirmation text naming the resulting model id.

        Raises:
            ValueError: when no provider is currently configured (propagated
                from ``settings.model_id``); ``model_name`` is restored to its
                previous value first.
        """
        settings = settings_manager.settings
        previous_model_name = settings.model_name
        settings.model_name = model_name
        try:
            model_id = settings.model_id
        except ValueError:
            settings.model_name = previous_model_name
            raise
        settings_manager.apply()
        settings_manager.save()
        application.set_displayed_model_id(model_id)
        return f"Model set to {model_id}"
