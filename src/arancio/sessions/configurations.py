"""Session-controlled configuration saved with each session."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from arancio.core.clients.configs.model import ModelConfig
from arancio.core.clients.configs.reasoning import ReasoningConfig
from arancio.core.permissions.types import PermissionCategory, PermissionLevel

if TYPE_CHECKING:
    from arancio.settings.base import Settings


@dataclass
class SessionConfiguration:
    """The session-controlled configuration active for one chat session.

    Attributes:
        model: the model section, the same Core type the settings use.
        permissions: the levels of the default permission categories.
    """

    model: ModelConfig
    permissions: dict[PermissionCategory, PermissionLevel]

    def to_dict(self) -> dict[str, Any]:
        """Render the configuration with JSON-compatible values.

        Returns:
            The nested model section and the permission mapping.
        """
        return {
            "model": self.model.to_dict(),
            "permissions": {
                category.name.lower(): level.value
                for category, level in self.permissions.items()
            },
        }

    def copy(self) -> SessionConfiguration:
        """Copy configuration values while retaining permission-category identities.

        Returns:
            An independent configuration and permission dictionary; the frozen
            model is shared.
        """
        return SessionConfiguration(
            model=self.model,
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
            model=settings.model,
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

        model = data.get("model")
        if not isinstance(model, Mapping):
            raise ValueError("configuration model is missing or invalid")
        reasoning = model.get("reasoning")
        if not isinstance(reasoning, Mapping):
            raise ValueError("configuration model.reasoning is missing or invalid")
        values = {
            "model.provider": model.get("provider"),
            "model.name": model.get("name"),
            "model.reasoning.effort": reasoning.get("effort"),
            "model.reasoning.summary": reasoning.get("summary"),
        }
        for path, value in values.items():
            if value is not None and not isinstance(value, str):
                raise ValueError(f"configuration {path} is invalid")

        return cls(
            model=ModelConfig(
                provider=values["model.provider"],
                name=values["model.name"],
                reasoning=ReasoningConfig(
                    effort=values["model.reasoning.effort"],
                    summary=values["model.reasoning.summary"],
                ),
            ),
            permissions=resolved_permissions,
        )
