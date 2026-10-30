"""User-configurable settings holder and its YAML-serializable form."""

from collections.abc import Collection, Mapping
from typing import Any

from arancio.core.permissions.types import PermissionCategory, PermissionLevel


class Settings:
    """In-memory snapshot of arancio's user-configurable settings.

    SettingsManager constructs and copies these objects after validation.
    This holder performs no provider discovery, validation or file I/O.
    :meth:`from_dict` populates an existing object from a complete checked mapping.

    Attributes:
        permissions: category to autonomy level mapping, covering every
            category; :attr:`~arancio.core.permissions.types.PermissionLevel.NONE`
            marks a category as not granted.
        provider: the agent model's provider prefix (e.g. ``"openai"``), or
            ``None`` when not yet configured. SettingsManager supplies the
            normalized, validated value.
        model_name: the agent model's name, without the provider prefix, or
            ``None`` when not yet configured. The web-summary client uses this
            same model.
        thinking_effort: the model's reasoning effort, or ``None`` to disable
            thinking entirely.
        thinking_summary: the model's reasoning summary mode, or ``None`` to
            disable summaries (e.g. for Ollama models).
        max_turns: the maximum number of agent turns per run, or ``"inf"``
            for no limit.
        max_retries: the maximum number of consecutive failed agent turns
            per run.
        turn_wait_time: the base wait, in seconds, before retrying a failed turn.
        turn_wait_time_multiplier: the factor the wait grows by on each
            consecutive retry.
    """

    def __init__(
        self,
        permissions: dict[PermissionCategory, PermissionLevel],
        provider: str | None,
        model_name: str | None,
        thinking_effort: str | None,
        thinking_summary: str | None,
        max_turns: int | str,
        max_retries: int,
        turn_wait_time: float,
        turn_wait_time_multiplier: float,
    ) -> None:
        """Initialize the settings snapshot.

        Args:
            permissions: category to autonomy level mapping, covering every
                category.
            provider: the checked provider prefix, or ``None`` when not configured.
            model_name: the agent model's name, or ``None`` when not yet
                configured.
            thinking_effort: the model's reasoning effort, or ``None`` to
                disable thinking entirely.
            thinking_summary: the model's reasoning summary mode, or ``None``
                to disable summaries.
            max_turns: the maximum number of agent turns per run, or ``"inf"``
                for no limit.
            max_retries: the maximum number of consecutive failed agent turns
                per run.
            turn_wait_time: the base wait, in seconds, before retrying a
                failed turn.
            turn_wait_time_multiplier: the factor the wait grows by on each
                consecutive retry.
        """
        self.permissions = permissions
        self.provider = provider
        self.model_name = model_name
        self.thinking_effort = thinking_effort
        self.thinking_summary = thinking_summary
        self.max_turns = max_turns
        self.max_retries = max_retries
        self.turn_wait_time = turn_wait_time
        self.turn_wait_time_multiplier = turn_wait_time_multiplier

    def __eq__(self, other: object) -> bool:
        """Compare settings by their field values.

        Args:
            other: the object to compare against.

        Returns:
            ``True`` when ``other`` is a :class:`Settings` with equal fields.
        """
        if not isinstance(other, Settings):
            return NotImplemented
        return (
            self.permissions == other.permissions
            and self.provider == other.provider
            and self.model_name == other.model_name
            and self.thinking_effort == other.thinking_effort
            and self.thinking_summary == other.thinking_summary
            and self.max_turns == other.max_turns
            and self.max_retries == other.max_retries
            and self.turn_wait_time == other.turn_wait_time
            and self.turn_wait_time_multiplier == other.turn_wait_time_multiplier
        )

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the settings.

        Returns:
            A compact string with every field's current value.
        """
        return (
            f"{type(self).__name__}("
            f"permissions={self.permissions!r}, "
            f"provider={self.provider!r}, "
            f"model_name={self.model_name!r}, "
            f"thinking_effort={self.thinking_effort!r}, "
            f"thinking_summary={self.thinking_summary!r}, "
            f"max_turns={self.max_turns!r}, "
            f"max_retries={self.max_retries!r}, "
            f"turn_wait_time={self.turn_wait_time!r}, "
            f"turn_wait_time_multiplier={self.turn_wait_time_multiplier!r}"
            ")"
        )

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

    def from_dict(
        self,
        data: Mapping[str, Any],
        categories: Collection[PermissionCategory],
    ) -> None:
        """Replace this object's fields from a complete, checked mapping.

        Args:
            data: the validated serialized values, as produced by :meth:`to_dict`.
            categories: the known permission category identities used for lookup.
        """
        categories_by_name = {
            category.name.lower(): category for category in categories
        }
        self.permissions = {
            categories_by_name[name.lower()]: PermissionLevel(level)
            for name, level in data["permissions"].items()
        }
        self.provider = data["provider"]
        self.model_name = data["model_name"]
        self.thinking_effort = data["thinking_effort"]
        self.thinking_summary = data["thinking_summary"]
        self.max_turns = data["max_turns"]
        self.max_retries = data["max_retries"]
        self.turn_wait_time = data["turn_wait_time"]
        self.turn_wait_time_multiplier = data["turn_wait_time_multiplier"]
