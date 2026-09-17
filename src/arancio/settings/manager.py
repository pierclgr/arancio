"""Settings facade: coordinate persistence and apply settings to live objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arancio.core.agents import Agent
from arancio.core.clients.base import BaseClient
from arancio.core.messages import Message
from arancio.core.permissions.types import PermissionCategory
from arancio.settings.settings import Settings
from arancio.settings.validator import SettingsValidator
from arancio.storage.manager import StorageManager

if TYPE_CHECKING:
    from arancio.sessions.session import SessionConfiguration


class SettingsManager:
    """Coordinate settings persistence and apply them to the live objects.

    The single settings surface the entry points use: it loads settings through
    the storage layer (creating defaults on first run), pushes them into the
    client, summary client (kept on the same model as the client) and agent, and
    persists changes back to disk.

    Attributes:
        _storage_manager: the storage layer reading and writing the settings file.
        _client: the agent's main client (model and thinking settings).
        _summary_client: the web-summary client (uses the main client's model).
        _agent: the agent (loop limits and permissions).
        _settings: the current in-memory settings, seeded with the defaults
            until :meth:`load` replaces them.
        _global_settings: the persisted defaults new sessions start from, seeded
            the same way.
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        client: BaseClient,
        summary_client: BaseClient,
        agent: Agent,
    ) -> None:
        """Initialize the manager with the storage layer and live objects.

        Args:
            storage_manager: the storage layer reading and writing the settings file.
            client: the agent's main client.
            summary_client: the web-summary client; kept on the same model as
                ``client``.
            agent: the agent whose loop limits and permissions are configured.
        """
        self._storage_manager = storage_manager
        self._client = client
        self._summary_client = summary_client
        self._agent = agent
        self._settings: Settings = Settings.default()
        self._global_settings: Settings = Settings.default()

    @property
    def settings(self) -> Settings:
        """Return the current in-memory settings.

        Returns:
            The current settings.
        """
        return self._settings

    @settings.setter
    def settings(self, value: Settings) -> None:
        """Replace the current in-memory settings without applying them.

        Args:
            value: the new settings; call :meth:`apply` and :meth:`save` to push
                and persist them.
        """
        self._settings = value

    @property
    def global_settings(self) -> Settings:
        """Return the persisted defaults used by newly created sessions.

        Returns:
            The global defaults.
        """
        return self._global_settings

    def load(self) -> tuple[Settings, list[Message]]:
        """Load, validate and apply the settings from disk.

        Creates the default settings file on first run (delegated to the
        storage layer, which also reports a missing or unreadable file). The
        storage layer signals that condition with ``None`` in place of a
        dictionary; deciding that ``None`` means "use the default settings"
        is this method's call, not the storage layer's — the default
        settings' own dictionary form is validated field by field the same
        way a real file's would be, so e.g. an unconfigured
        ``provider``/``model_name`` is still reported. Otherwise, the parsed
        dictionary is validated by
        :class:`~arancio.settings.validator.SettingsValidator`, which falls
        back to defaults for any missing or invalid field. The loaded
        settings are then applied to the live objects.

        Returns:
            A ``(settings, messages)`` pair: the loaded settings, and the
            file-level and field-level messages to surface in the UI.
        """
        data, file_messages = self._storage_manager.load_settings()
        if data is None:
            data = Settings.default().to_dict()
        self._global_settings, field_messages = SettingsValidator.validate(data)
        self._settings = self._copy_settings(self._global_settings)
        self.apply()
        return self._settings, file_messages + field_messages

    def save(self) -> None:
        """Persist the current settings to disk through the storage layer."""
        self._storage_manager.save_settings(self._settings)
        self._global_settings = self._copy_settings(self._settings)

    def save_provider(self) -> None:
        """Persist only the active provider as a global default."""
        self._global_settings.provider = self._settings.provider
        self._storage_manager.save_settings(self._global_settings)

    def save_model_name(self) -> None:
        """Persist only the active model name as a global default."""
        self._global_settings.model_name = self._settings.model_name
        self._storage_manager.save_settings(self._global_settings)

    def save_thinking_effort(self) -> None:
        """Persist only the active thinking effort as a global default."""
        self._global_settings.thinking_effort = self._settings.thinking_effort
        self._storage_manager.save_settings(self._global_settings)

    def save_permission(self, category: PermissionCategory) -> None:
        """Persist one active permission category as a global default.

        Args:
            category: the permission category whose active level to persist.
        """
        self._global_settings.permissions[category] = self._settings.permissions[
            category
        ]
        self._storage_manager.save_settings(self._global_settings)

    def apply_session_configuration(self, configuration: SessionConfiguration) -> None:
        """Apply a restored session's command-controlled configuration.

        Args:
            configuration: an object exposing provider, model_name,
                thinking_effort and permissions from a saved session.
        """
        settings = self._copy_settings(self._global_settings)
        settings.provider = configuration.provider
        settings.model_name = configuration.model_name
        settings.thinking_effort = configuration.thinking_effort
        settings.permissions = dict(configuration.permissions)
        self._settings = settings
        self.apply()

    def reset_to_global(self) -> None:
        """Restore the active runtime configuration from global defaults."""
        self._settings = self._copy_settings(self._global_settings)
        self.apply()

    def apply(self) -> None:
        """Push the current settings into the live objects.

        Sets the client's model and thinking settings, the summary client's model (the
        same model as the main client), the agent's loop limits, and replaces the
        agent's permission grants (which rebuilds its tool catalog). The summary
        client's thinking is left untouched: it is disabled once where the summary
        client is constructed. The client's model is left ``None`` while the provider or
        model name is not yet configured.
        """
        settings = self._settings
        model_id = (
            settings.model_id if settings.provider and settings.model_name else None
        )
        self._client.model_id = model_id
        self._client.thinking_effort = settings.thinking_effort
        self._client.thinking_summary = settings.thinking_summary
        self._summary_client.model_id = model_id
        self._agent.max_turns = settings.max_turns
        self._agent.max_retries = settings.max_retries
        self._agent.retry_delay = settings.turn_wait_time
        self._agent.retry_delay_multiplier = settings.turn_wait_time_multiplier
        self._agent.set_permissions(settings.permissions)

    @staticmethod
    def _copy_settings(settings: Settings) -> Settings:
        """Return an independent settings copy from its serialized form.

        Args:
            settings: the settings snapshot to copy.

        Returns:
            An independent settings object with equal values.
        """
        return Settings.from_dict(settings.to_dict())
