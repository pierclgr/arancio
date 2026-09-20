"""Provider command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.commands.state_change import StateChangeCommand
from arancio.core.messages import ErrorMessage

if TYPE_CHECKING:
    from arancio.sessions.manager import SessionManager
    from arancio.settings.manager import SettingsManager
    from arancio.ui.app import App


class ProviderCommand(StateChangeCommand):
    """Command that sets the provider, keeping the current model name."""

    name = "provider"
    description = "Set the provider, keeping the current model name."

    @classmethod
    def execute(
        cls,
        provider: str,
        application: App,
        settings_manager: SettingsManager,
        session_manager: SessionManager,
    ) -> str | ErrorMessage:
        """Replace the provider and apply it.

        Unlike :class:`~arancio.commands.model.ModelCommand`, this does not
        require a model name to already be configured — setting the provider
        alone is the first configuration step on a fresh install. Assigning
        ``settings.provider`` below raises ``ValueError`` before anything is
        persisted when the value is not a valid LiteLLM provider name.

        Args:
            provider: the new provider prefix (e.g. ``"openai"``), bound to
                the prompt's first word, matched case-insensitively against
                LiteLLM's supported providers. The current model name is kept
                unchanged.
            application: the running app whose toolbar is refreshed with the
                new model id, when a model name is already configured too.
            settings_manager: the manager used to apply and persist the
                change.
            session_manager: the active session manager, whose current
                session's configuration is updated to match and whose
                recorder persists the change.

        Returns:
            Confirmation text naming the new provider, or the persistence
            error notice when saving the change failed.
        """
        settings = settings_manager.settings
        settings.provider = provider
        settings_manager.apply()
        settings_manager.save_provider()
        if settings.model_name:
            application.set_displayed_model_id(settings.model_id)
        cls._apply_configuration(session_manager, settings_manager)
        return cls._persist_state_change(
            session_manager, f"Provider set to {settings.provider}"
        )
