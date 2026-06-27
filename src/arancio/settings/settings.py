"""User-configurable settings holder and its YAML-serializable form."""

from dataclasses import dataclass
from typing import Any

from arancio.core.constants.agent import (
    AGENT_DEFAULT_MAX_TURNS,
    AGENT_DEFAULT_TURN_WAIT_TIME,
    AGENT_DEFAULT_TURN_WAIT_TIME_MULTIPLIER,
)
from arancio.core.constants.litellm import (
    LITELLM_DEFAULT_THINKING_EFFORT,
    LITELLM_DEFAULT_THINKING_SUMMARY,
)
from arancio.core.types.permissions import PermissionCategory, PermissionLevel


@dataclass
class Settings:
    """In-memory snapshot of arancio's user-configurable settings.

    A plain data holder produced from disk by the storage layer and pushed into
    the live objects by the settings manager. :meth:`default` builds it from the
    registered code defaults, so the generated default file matches the values
    the code itself uses.

    Attributes:
        permissions: granted category to autonomy level mapping.
        model_id: the agent's model id, or ``None`` when not yet configured.
        summary_model_id: the web-summary model id, or ``None`` when not yet
            configured.
        thinking_effort: the model's reasoning effort.
        thinking_summary: the model's reasoning summary mode, or ``None`` to
            disable summaries (e.g. for Ollama models).
        max_turns: the maximum number of agent turns per run.
        turn_wait_time: the base wait, in seconds, before retrying a failed turn.
        turn_wait_time_multiplier: the factor the wait grows by on each
            consecutive retry.
    """

    permissions: dict[PermissionCategory, PermissionLevel]
    model_id: str | None
    summary_model_id: str | None
    thinking_effort: str
    thinking_summary: str | None
    max_turns: int
    turn_wait_time: float
    turn_wait_time_multiplier: float

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
            model_id=None,
            summary_model_id=None,
            thinking_effort=LITELLM_DEFAULT_THINKING_EFFORT,
            thinking_summary=LITELLM_DEFAULT_THINKING_SUMMARY,
            max_turns=AGENT_DEFAULT_MAX_TURNS,
            turn_wait_time=AGENT_DEFAULT_TURN_WAIT_TIME,
            turn_wait_time_multiplier=AGENT_DEFAULT_TURN_WAIT_TIME_MULTIPLIER,
        )

    def to_dict(self) -> dict[str, Any]:
        """Render the settings as a YAML-serializable dictionary.

        Permissions are flattened to plain strings (category name to level
        value) so the file stays human-readable and free of Python objects.

        Returns:
            A dictionary with only built-in types, ready for ``yaml.safe_dump``.
        """
        return {
            "permissions": {
                category.name: level.value
                for category, level in self.permissions.items()
            },
            "model_id": self.model_id,
            "summary_model_id": self.summary_model_id,
            "thinking_effort": self.thinking_effort,
            "thinking_summary": self.thinking_summary,
            "max_turns": self.max_turns,
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
                PermissionCategory[name]: PermissionLevel(level)
                for name, level in data["permissions"].items()
            },
            model_id=data["model_id"],
            summary_model_id=data["summary_model_id"],
            thinking_effort=data["thinking_effort"],
            thinking_summary=data["thinking_summary"],
            max_turns=data["max_turns"],
            turn_wait_time=data["turn_wait_time"],
            turn_wait_time_multiplier=data["turn_wait_time_multiplier"],
        )
