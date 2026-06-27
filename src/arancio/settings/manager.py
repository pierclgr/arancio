"""Settings facade: coordinate persistence and apply settings to live objects."""

from arancio.core.agents import Agent
from arancio.core.clients.base import BaseClient
from arancio.settings.settings import Settings
from arancio.storage.manager import StorageManager


class SettingsManager:
    """Coordinate settings persistence and apply them to the live objects.

    The single settings surface the entry points use: it loads settings through
    the storage layer (creating defaults on first run), pushes them into the
    client, summary client and agent, and persists changes back to disk.

    Attributes:
        _storage: the storage layer reading and writing the settings file.
        _client: the agent's main client (model and thinking settings).
        _summary_client: the web-summary client (summary model).
        _agent: the agent (loop limits and permissions).
        _settings: the current in-memory settings, or ``None`` before the first
            load.
    """

    def __init__(
        self,
        storage: StorageManager,
        client: BaseClient,
        summary_client: BaseClient,
        agent: Agent,
    ) -> None:
        """Initialize the manager with the storage layer and live objects.

        Args:
            storage: the storage layer reading and writing the settings file.
            client: the agent's main client.
            summary_client: the web-summary client.
            agent: the agent whose loop limits and permissions are configured.
        """
        self._storage = storage
        self._client = client
        self._summary_client = summary_client
        self._agent = agent
        self._settings: Settings | None = None

    @property
    def settings(self) -> Settings | None:
        """Return the current in-memory settings.

        Returns:
            The current settings, or ``None`` before the first load.
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

    def load(self) -> Settings:
        """Load the settings from disk and apply them to the live objects.

        Creates the default settings file on first run (delegated to the storage
        layer), holds the loaded settings, then applies them.

        Returns:
            The loaded settings.
        """
        self._settings = self._storage.load_settings()
        self.apply()
        return self._settings

    def save(self) -> None:
        """Persist the current settings to disk through the storage layer."""
        self._storage.save_settings(self._settings)

    def apply(self) -> None:
        """Push the current settings into the live objects.

        Sets the client's model and thinking settings, the summary client's model, the
        agent's loop limits, and replaces the agent's permission grants (which rebuilds
        its tool catalog). The summary client's thinking is left untouched: it is
        disabled once where the summary client is constructed.
        """
        settings = self._settings
        self._client.model_id = settings.model_id
        self._client.thinking_effort = settings.thinking_effort
        self._client.thinking_summary = settings.thinking_summary
        self._summary_client.model_id = settings.summary_model_id
        self._agent.max_turns = settings.max_turns
        self._agent.retry_delay = settings.turn_wait_time
        self._agent.retry_delay_multiplier = settings.turn_wait_time_multiplier
        self._agent.set_permissions(settings.permissions)
