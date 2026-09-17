"""In-memory models for one persistent chat session."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from arancio.core.permissions.types import PermissionCategory, PermissionLevel

if TYPE_CHECKING:
    from arancio.settings.settings import Settings


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
    def from_dict(cls, data: dict[str, Any]) -> SessionConfiguration:
        """Rebuild a session configuration from one JSON record.

        Args:
            data: the JSON-compatible configuration mapping.

        Returns:
            The reconstructed session configuration.

        Raises:
            ValueError: when a required field is missing or invalid.
        """
        permissions = data.get("permissions")
        if not isinstance(permissions, dict):
            raise ValueError("configuration permissions are missing or invalid")

        try:
            resolved_permissions = {
                PermissionCategory[name.upper()]: PermissionLevel(level)
                for name, level in permissions.items()
            }
        except (KeyError, ValueError) as exc:
            raise ValueError("configuration permissions are invalid") from exc

        if set(resolved_permissions) != set(PermissionCategory):
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
    """One chat session and the events needed to restore it."""

    id: str
    name: str
    created_at: datetime
    creation_working_directory: Path
    working_directory: Path
    path: Path
    configuration: SessionConfiguration
    events: list[SessionEvent] = field(default_factory=list)
    recovery_offset: int | None = None
    created_on_disk: bool = True
    file_states: dict[str, float] = field(default_factory=dict)

    def add_event(self, record: dict[str, Any], saved: bool = False) -> SessionEvent:
        """Append one event to this session's ordered log.

        Args:
            record: the JSON-compatible event record.
            saved: whether the event is already durable on disk.

        Returns:
            The newly appended in-memory event.
        """
        event = SessionEvent(record=record, saved=saved)
        self.events.append(event)
        return event
