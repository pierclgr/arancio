"""User-configurable settings holder and its YAML-serializable form."""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from arancio.core.clients.configs.model import ModelConfig
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
        modl: the agent model's provider, name and reasoning options.
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
        plugins: dict[str, PluginSettings],
        model: ModelConfig,
        max_turns: int | str,
        max_retries: int,
        turn_wait_time: float,
        turn_wait_time_multiplier: float,
    ) -> None:
        """Initialize the settings snapshot.

        Args:
            permissions: default category to autonomy level mapping, covering
                every default category.
            plugins: lowercase plugin names mapped to their settings entries.
            model: the agent model's checked provider, name and reasoning options.
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
        self.plugins = plugins
        self.model = model
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
            and self.plugins == other.plugins
            and self.model == other.model
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
            f"plugins={self.plugins!r}, "
            f"model={self.model!r}, "
            f"max_turns={self.max_turns!r}, "
            f"max_retries={self.max_retries!r}, "
            f"turn_wait_time={self.turn_wait_time!r}, "
            f"turn_wait_time_multiplier={self.turn_wait_time_multiplier!r}"
            ")"
        )

    def to_dict(self) -> dict[str, Any]:
        """Render the settings as a YAML-serializable dictionary.

        Permissions are flattened to plain strings (lowercase category name to
        level value) under ``default``, plugins to a list of one-key mappings, and
        the model to a ``model`` section with a nested ``reasoning`` section, so the
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
        self.model = ModelConfig.from_dict(data["model"])
        self.max_turns = data["max_turns"]
        self.max_retries = data["max_retries"]
        self.turn_wait_time = data["turn_wait_time"]
        self.turn_wait_time_multiplier = data["turn_wait_time_multiplier"]
