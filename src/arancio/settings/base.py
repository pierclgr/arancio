"""User-configurable settings holder and its YAML-serializable form."""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from arancio.core.agents.types import Turn
from arancio.core.clients.types import Model
from arancio.core.permissions.types import PermissionCategory, PermissionLevel


@dataclass
class PluginSettings:
    """One entry of the plugins settings.

    Attributes:
        enabled: whether startup loads the plugin.
        permission_level: the level of the plugin's ``PLUGIN:<name>`` category,
            used by its ``CUSTOM`` tools.
    """

    enabled: bool
    permission_level: PermissionLevel


class Settings:
    """In-memory snapshot of arancio's user-configurable settings.

    SettingsManager constructs and copies these objects after validation.
    This holder performs no provider discovery, validation or file I/O.
    :meth:`from_dict` populates an existing object from a complete checked mapping.

    Attributes:
        permissions: default category to autonomy level mapping, covering
            every default category;
            :attr:`~arancio.core.permissions.types.PermissionLevel.NONE` marks a
            category as not granted. Saved under ``permissions.default``.
        plugins: lowercase plugin names mapped to their settings entries, in file
            order. A plugin without an entry is not loaded.
        model: the agent model's provider, name and reasoning options.
        turn: the agent loop's turn limit and retry rules for failed turns.
    """

    def __init__(
        self,
        permissions: dict[PermissionCategory, PermissionLevel],
        plugins: dict[str, PluginSettings],
        model: Model,
        turn: Turn,
    ) -> None:
        """Initialize the settings snapshot.

        Args:
            permissions: default category to autonomy level mapping, covering
                every default category.
            plugins: lowercase plugin names mapped to their settings entries.
            model: the agent model's checked provider, name and reasoning options.
            turn: the agent loop's checked turn limit and retry rules.
        """
        self.permissions = permissions
        self.plugins = plugins
        self.model = model
        self.turn = turn

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
            and self.plugins == other.plugins
            and self.model == other.model
            and self.turn == other.turn
        )

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the settings.

        Returns:
            A compact string with every field's current value.
        """
        return (
            f"{type(self).__name__}("
            f"permissions={self.permissions!r}, "
            f"plugins={self.plugins!r}, "
            f"model={self.model!r}, "
            f"turn={self.turn!r}"
            ")"
        )

    def to_dict(self) -> dict[str, Any]:
        """Render the settings as a YAML-serializable dictionary.

        Permissions are flattened to plain strings (lowercase category name to
        level value) under ``default``, plugins to a list of one-key mappings, the
        model to a ``model`` section with a nested ``reasoning`` section, and the
        agent loop to a ``turn`` section with a nested ``retry`` section, so the
        file stays human-readable and free of Python objects.

        Returns:
            A dictionary with only built-in types, ready for ``yaml.safe_dump``.
        """
        return {
            "permissions": {
                "default": {
                    category.name.lower(): level.value
                    for category, level in self.permissions.items()
                },
            },
            "plugins": [
                {
                    name: {
                        "enabled": plugin.enabled,
                        "permission_level": plugin.permission_level.value,
                    }
                }
                for name, plugin in self.plugins.items()
            ],
            "model": self.model.to_dict(),
            "turn": self.turn.to_dict(),
        }

    def from_dict(
        self,
        data: Mapping[str, Any],
        categories: Collection[PermissionCategory],
    ) -> None:
        """Replace this object's fields from a complete, checked mapping.

        Args:
            data: the validated serialized values, as produced by :meth:`to_dict`.
            categories: the default permission category identities used for lookup.
        """
        categories_by_name = {
            category.name.lower(): category for category in categories
        }
        self.permissions = {
            categories_by_name[name.lower()]: PermissionLevel(level)
            for name, level in data["permissions"]["default"].items()
        }
        self.plugins = {}
        for entry in data["plugins"]:
            for name, values in entry.items():
                self.plugins[name.lower()] = PluginSettings(
                    enabled=values["enabled"],
                    permission_level=PermissionLevel(values["permission_level"]),
                )
        self.model = Model.from_dict(data["model"])
        self.turn = Turn.from_dict(data["turn"])
