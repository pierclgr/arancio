"""User-configurable settings holder and its YAML-serializable form."""

from dataclasses import dataclass
from typing import Any

from arancio.core.constants.agent import (
    AGENT_DEFAULT_MAX_RETRIES,
    AGENT_DEFAULT_MAX_TURNS,
    AGENT_DEFAULT_TURN_WAIT_TIME,
    AGENT_DEFAULT_TURN_WAIT_TIME_MULTIPLIER,
)
from arancio.core.constants.litellm import (
    LITELLM_DEFAULT_THINKING_EFFORT,
    LITELLM_DEFAULT_THINKING_SUMMARY,
)
from arancio.core.permissions.types import PermissionCategory, PermissionLevel


@dataclass
class Settings:
    """In-memory snapshot of arancio's user-configurable settings.

    A plain data holder produced from disk by the storage layer and pushed into
    the live objects by the settings manager. :meth:`default` builds it from the
    registered code defaults, so the generated default file matches the values
    the code itself uses.

    Attributes:
        permissions: granted category to autonomy level mapping.
        provider: the agent model's provider prefix (e.g. ``"openai"``), or
            ``None`` when not yet configured.
        model_name: the agent model's name, without the provider prefix, or
            ``None`` when not yet configured. The web-summary client uses this
            same model.
        thinking_effort: the model's reasoning effort, or ``None`` to disable
            thinking entirely.
        thinking_summary: the model's reasoning summary mode, or ``None`` to
            disable summaries (e.g. for Ollama models).
        max_turns: the maximum number of agent turns per run, or ``None``
            for no limit.
        max_retries: the maximum number of consecutive failed agent turns
            per run.
        turn_wait_time: the base wait, in seconds, before retrying a failed turn.
        turn_wait_time_multiplier: the factor the wait grows by on each
            consecutive retry.
    """

    permissions: dict[PermissionCategory, PermissionLevel]
    provider: str | None
    model_name: str | None
    thinking_effort: str | None
    thinking_summary: str | None
    max_turns: int | None
    max_retries: int
    turn_wait_time: float
    turn_wait_time_multiplier: float

    @property
    def model_id(self) -> str:
        """Build the model id from the current provider and model name.

        Returns:
            The joined ``provider/model_name`` model id.

        Raises:
            ValueError: when the provider is not configured.
            ValueError: when the model name is not configured.
        """
        if not self.provider:
            raise ValueError("No provider configured.")
        if not self.model_name:
            raise ValueError("No model name configured.")
        return f"{self.provider}/{self.model_name}"

    @classmethod
    def default(cls) -> "Settings":
        """Build settings from the registered code defaults.

        Returns:
            A settings snapshot with every category granted at
            :attr:`PermissionLevel.ASK`, unconfigured models and the registered
            thinking and agent-loop defaults.
        """
        return cls(
            permissions={
                category: PermissionLevel.ASK for category in PermissionCategory
            },
            provider=None,
            model_name=None,
            thinking_effort=LITELLM_DEFAULT_THINKING_EFFORT,
            thinking_summary=LITELLM_DEFAULT_THINKING_SUMMARY,
            max_turns=AGENT_DEFAULT_MAX_TURNS,
            max_retries=AGENT_DEFAULT_MAX_RETRIES,
            turn_wait_time=AGENT_DEFAULT_TURN_WAIT_TIME,
            turn_wait_time_multiplier=AGENT_DEFAULT_TURN_WAIT_TIME_MULTIPLIER,
        )

    def to_dict(self) -> dict[str, Any]:
        """Render the settings as a YAML-serializable dictionary.

        Permissions are flattened to plain strings (lowercase category name to
        level value) so the file stays human-readable and free of Python
        objects.

        Returns:
            A dictionary with only built-in types, ready for ``yaml.safe_dump``.
        """
        return {
            "permissions": {
                category.name.lower(): level.value
                for category, level in self.permissions.items()
            },
            "provider": self.provider,
            "model_name": self.model_name,
            "thinking_effort": self.thinking_effort,
            "thinking_summary": self.thinking_summary,
            "max_turns": self.max_turns,
            "max_retries": self.max_retries,
            "turn_wait_time": self.turn_wait_time,
            "turn_wait_time_multiplier": self.turn_wait_time_multiplier,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        """Rebuild settings from a dictionary loaded from the YAML file.

        Args:
            data: the dictionary parsed from disk, as produced by
                :meth:`to_dict`.

        Returns:
            The reconstructed settings, with permission strings resolved back to
            :class:`PermissionCategory` and :class:`PermissionLevel` members.
        """
        return cls(
            permissions={
                PermissionCategory[name.upper()]: PermissionLevel(level)
                for name, level in data["permissions"].items()
            },
            provider=data["provider"],
            model_name=data["model_name"],
            thinking_effort=data["thinking_effort"],
            thinking_summary=data["thinking_summary"],
            max_turns=data["max_turns"],
            max_retries=data["max_retries"],
            turn_wait_time=data["turn_wait_time"],
            turn_wait_time_multiplier=data["turn_wait_time_multiplier"],
        )
