"""In-memory models for one persistent chat session."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from arancio.core.permissions.types import PermissionCategory, PermissionLevel

if TYPE_CHECKING:
    from arancio.settings.base import Settings


@dataclass
class SessionConfiguration:
    """The command-controlled configuration active for one chat session."""

    provider: str | None
    model_name: str | None
    thinking_effort: str | None
    permissions: dict[PermissionCategory, PermissionLevel]

    def to_dict(self) -> dict[str, Any]:
        """Render the configuration with JSON-compatible values.

        Returns:
            The provider, model, thinking effort and permission mapping.
        """
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "thinking_effort": self.thinking_effort,
            "permissions": {
                category.name.lower(): level.value
                for category, level in self.permissions.items()
            },
        }

    def copy(self) -> SessionConfiguration:
        """Copy configuration values while retaining permission-category identities.

        Returns:
            An independent configuration and permission dictionary.
        """
        return SessionConfiguration(
            provider=self.provider,
            model_name=self.model_name,
            thinking_effort=self.thinking_effort,
            permissions=dict(self.permissions),
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> SessionConfiguration:
        """Copy command-controlled values from the active application settings.

        Args:
            settings: the active settings object supplied by ``SettingsManager``.

        Returns:
            A detached session configuration snapshot.
        """
        return cls(
            provider=settings.provider,
            model_name=settings.model_name,
            thinking_effort=settings.thinking_effort,
            permissions=dict(settings.permissions),
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        categories: Collection[PermissionCategory],
    ) -> SessionConfiguration:
        """Rebuild a session configuration from one JSON record.

        Args:
            data: the JSON-compatible configuration mapping.
            categories: the known permission-category identities.

        Returns:
            The reconstructed session configuration.

        Raises:
            ValueError: when a required field is missing or invalid.
        """
        permissions = data.get("permissions")
        if not isinstance(permissions, dict):
            raise ValueError("configuration permissions are missing or invalid")

        known_categories = {category.name.lower(): category for category in categories}
        resolved_permissions: dict[PermissionCategory, PermissionLevel] = {}
        for name, level in permissions.items():
            if not isinstance(name, str) or name.lower() not in known_categories:
                raise ValueError(f"Unknown configuration permission category: {name!r}")
            if level is not None and not isinstance(level, str):
                raise ValueError(f"Invalid configuration permission level: {level!r}")
            try:
                resolved_level = PermissionLevel(level)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid configuration permission level: {level!r}"
                ) from exc
            resolved_permissions[known_categories[name.lower()]] = resolved_level

        if set(resolved_permissions) != set(categories):
            raise ValueError("configuration must contain every permission category")

        provider = data.get("provider")
        model_name = data.get("model_name")
        thinking_effort = data.get("thinking_effort")
        if provider is not None and not isinstance(provider, str):
            raise ValueError("configuration provider is invalid")
        if model_name is not None and not isinstance(model_name, str):
            raise ValueError("configuration model_name is invalid")
        if thinking_effort is not None and not isinstance(thinking_effort, str):
            raise ValueError("configuration thinking_effort is invalid")

        return cls(
            provider=provider,
            model_name=model_name,
            thinking_effort=thinking_effort,
            permissions=resolved_permissions,
        )


@dataclass
class SessionEvent:
    """One JSONL event and its in-process persistence state."""

    record: dict[str, Any]
    saved: bool = True


@dataclass
class Session:
    """One managed chat session and the events needed to restore it.

    A session forked from another carries the source session's ID in ``forked_from``; it
    is ``None`` for a session created or resumed normally.
    """

    id: str
    created_at: datetime
    creation_working_directory: Path
    working_directory: Path
    path: Path
    configuration: SessionConfiguration
    explicit_name: str | None = None
    forked_from: str | None = None
    events: list[SessionEvent] = field(default_factory=list)
    recovery_offset: int | None = None
    created_on_disk: bool = True
    file_states: dict[str, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Return the explicit name, or the ID when no name is set."""
        return self.id if self.explicit_name is None else self.explicit_name

    @name.setter
    def name(self, value: str | None) -> None:
        """Set the explicit name, or clear it with ``None``.

        Args:
            value: the explicit name to store.
        """
        self.explicit_name = value

    def add_event(self, record: dict[str, Any], saved: bool = False) -> SessionEvent:
        """Append one event to this session's ordered log.

        Args:
            record: the JSON-compatible event record.
            saved: whether the event is already durable on disk.

        Returns:
            The newly appended in-memory event.
        """
        event = SessionEvent(record=deepcopy(record), saved=saved)
        self.events.append(event)
        return event


@dataclass
class SessionState:
    """Detached current state without the managed event list or retry offset.

    Attributes:
        id: the conversation identity.
        created_at: the creation timestamp.
        creation_working_directory: the original working directory.
        working_directory: the current working directory.
        path: the fixed log location.
        configuration: an independent configuration snapshot.
        explicit_name: the chosen name, or None for the ID fallback.
        forked_from: the source identity, when forked.
        file_states: an independent saved path-to-mtime mapping.
        created_on_disk: whether the creation record was saved.
        has_pending_events: whether any records remain unsaved.
    """

    id: str
    created_at: datetime
    creation_working_directory: Path
    working_directory: Path
    path: Path
    configuration: SessionConfiguration
    explicit_name: str | None
    forked_from: str | None
    file_states: dict[str, float]
    created_on_disk: bool
    has_pending_events: bool

    @property
    def name(self) -> str:
        """Return the explicit name, or the ID when no name is set."""
        return self.id if self.explicit_name is None else self.explicit_name


@dataclass
class SessionRegistryEntry:
    """Search metadata and discovery status for one saved session file."""

    id: str
    working_directory: Path | None
    configuration: SessionConfiguration | None
    log_path: Path
    status: Literal["healthy", "unverified", "damaged"]
    explicit_name: str | None = None
    damage_reason: str | None = None

    @property
    def name(self) -> str:
        """Return the explicit name, or the ID when no name is set."""
        return self.id if self.explicit_name is None else self.explicit_name


@dataclass
class SessionScan:
    """Temporary validator result used to build a registry entry."""

    explicit_name: str | None
    working_directory: Path | None
    configuration: SessionConfiguration | None
    status: Literal["healthy", "unverified", "damaged"]
    damage_reason: str | None = None
