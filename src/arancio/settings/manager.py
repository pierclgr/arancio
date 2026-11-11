"""Settings values, validation and YAML persistence through generic Storage."""

from collections.abc import Collection, Mapping
from typing import Any

import yaml

from arancio.core.agents.constants import (
    AGENT_DEFAULT_MAX_RETRIES,
    AGENT_DEFAULT_MAX_TURNS,
    AGENT_DEFAULT_TURN_WAIT_TIME,
    AGENT_DEFAULT_TURN_WAIT_TIME_MULTIPLIER,
)
from arancio.core.clients.constants import (
    LITELLM_DEFAULT_THINKING_EFFORT,
    LITELLM_DEFAULT_THINKING_SUMMARY,
)
from arancio.core.messages.base import ErrorMessage, Message, WarningMessage
from arancio.core.permissions.constants import DEFAULT_PERMISSION_LEVELS
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.settings.base import PluginSettings, Settings
from arancio.settings.constants import ARANCIO_SETTINGS_FILENAME
from arancio.settings.validator import SettingsValidator
from arancio.storage.manager import StorageManager


class SettingsManager:
    """Own active settings and global defaults without applying them to Core.

    Only this facade constructs and copies Settings objects. Callers edit active
    values through its methods and read detached snapshots through properties.
    File policy belongs here; Storage supplies UTF-8 reads and writes.

    Attributes:
        __storage_manager: the injected generic file I/O service.
        __categories: the default permission categories, saved under
            ``permissions.default`` and stored in sessions.
        __validator: shared validation for file loading and explicit edits.
        __settings: the active conversation values.
        __global: the defaults selected by load or a successful save.
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        supported_providers: Collection[str],
    ) -> None:
        """Initialize defaults and copy metadata without reading or writing files.

        Args:
            storage_manager: the service reading and writing relative settings paths.
            supported_providers: the provider names available to the application.
        """
        self.__storage_manager = storage_manager
        self.__categories: tuple[PermissionCategory, ...] = tuple(
            DEFAULT_PERMISSION_LEVELS
        )
        self.__global: Settings = self.__default_settings()
        self.__settings: Settings = self.__copy(self.__global)
        self.__validator = SettingsValidator(
            self.__global.to_dict(), supported_providers
        )

    @property
    def settings(self) -> Settings:
        """A detached copy of the active settings.

        Returns:
            A snapshot whose fields and permission dictionary can be changed
            without changing manager-owned values.
        """
        return self.__copy(self.__settings)

    @property
    def global_settings(self) -> Settings:
        """A detached copy of the loaded or successfully saved defaults.

        Returns:
            An independent snapshot of the global-default values.
        """
        return self.__copy(self.__global)

    def load(self) -> tuple[Settings, list[Message]]:
        """Load settings through Storage and return file and field diagnostics.

        A missing file is created from code defaults. Empty, malformed or
        non-mapping YAML is left unchanged while defaults are used in memory.
        Other I/O and UTF-8 decoding failures propagate without publishing values.

        Returns:
            A detached active snapshot and the notices produced by loading.
        """
        defaults = self.__default_settings()
        file_messages: list[Message] = []
        try:
            text = self.__storage_manager.read_text(ARANCIO_SETTINGS_FILENAME)
        except FileNotFoundError:
            data = defaults.to_dict()
            self.__storage_manager.write_file(
                ARANCIO_SETTINGS_FILENAME, yaml.safe_dump(data, sort_keys=False)
            )
            file_messages.append(
                WarningMessage(
                    content=(
                        f"{ARANCIO_SETTINGS_FILENAME} not found; "
                        "creating it with default settings."
                    )
                )
            )
        else:
            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError:
                data = None
            if not isinstance(data, Mapping):
                data = defaults.to_dict()
                file_messages.append(
                    ErrorMessage(
                        content=(
                            f"{ARANCIO_SETTINGS_FILENAME} is empty or not a valid "
                            "YAML mapping; using default settings."
                        )
                    )
                )

        checked, field_messages = self.__validator.validate(data)
        defaults.from_dict(checked, self.__categories)
        active = self.__copy(defaults)
        self.__global = defaults
        self.__settings = active
        return self.settings, file_messages + field_messages

    def set_field(
        self,
        field: str,
        value: Any,
    ) -> None:
        """Validate and update one active setting without saving it.

        Permissions are merged rather than replaced. A model name must form a
        usable model ID, and changing thinking effort requires a configured model.

        Args:
            field: the setting to change.
            value: the new value, or typed category-to-level changes for permissions.

        Raises:
            ValueError: when the field or value is invalid, or a model is required.
        """
        data = self.__settings.to_dict()
        if not isinstance(field, str) or field not in data:
            raise ValueError(f"Unknown editable settings field: {field!r}.")
        if field == "thinking_effort":
            self.__settings.model_id

        if field == "permissions":
            data[field]["default"].update(self.__permission_values(value))
        else:
            data[field] = value

        candidate = self.__prepare(data)
        if field == "model_name":
            candidate.model_id
        self.__settings = candidate

    def restore_active(
        self,
        *,
        provider: str | None,
        model_name: str | None,
        thinking_effort: str | None,
        permissions: Mapping[PermissionCategory, PermissionLevel],
    ) -> None:
        """Restore four session-controlled values on top of global defaults.

        Thinking summary and loop limits retain their global-default values.
        This operation does not save settings or receive a Sessions object.
        Invalid values or incomplete permissions raise ValueError.

        Args:
            provider: the restored provider, or None when unconfigured.
            model_name: the restored model name, or None when unconfigured.
            thinking_effort: the restored effort, or None when disabled.
            permissions: the complete restored permission map for default categories.
        """
        values = self.__permission_values(permissions)
        data = self.__global.to_dict()
        data.update(
            provider=provider,
            model_name=model_name,
            thinking_effort=thinking_effort,
            permissions={"default": values},
        )
        self.__settings = self.__prepare(data)

    def reset_global(self) -> None:
        """Replace active values with global defaults without file I/O."""
        self.__settings = self.__copy(self.__global)

    def reset_default(self) -> None:
        """Write code defaults, then replace both active and global settings.

        Write failures propagate and leave both in-memory snapshots unchanged.
        """
        candidate = self.__default_settings()
        self.save(candidate)
        self.__settings = candidate

    def save_field(
        self, field: str, category: PermissionCategory | None = None
    ) -> None:
        """Persist one active field or permission category as a global default.

        Write failures propagate without changing global defaults.

        Args:
            field: the settings field whose active value to save.
            category: the known category to save when field is permissions.

        Raises:
            ValueError: when the field or category is invalid.
        """
        if not isinstance(field, str) or field not in self.__settings.to_dict():
            raise ValueError(f"Unknown settings field: {field!r}.")
        if field == "permissions" and category not in self.__categories:
            raise ValueError(f"Unknown permission category: {category!r}.")
        if field != "permissions" and category is not None:
            raise ValueError(f"Category is only valid for permissions: {category!r}.")

        candidate = self.__copy(self.__global)
        if field == "permissions":
            candidate.permissions[category] = self.__settings.permissions[category]
        else:
            setattr(candidate, field, getattr(self.__settings, field))
        self.save(candidate)

    def set_plugin_permission_levels(
        self, levels: Mapping[str, PermissionLevel]
    ) -> None:
        """Validate and update plugin permission levels without saving them.

        An invalid mapping or level, or a plugin without a settings entry, raises
        ValueError and changes nothing.

        Args:
            levels: plugin names, matched case-insensitively, mapped to typed levels.
        """
        candidate = self.__copy(self.__settings)
        for name, level in self.__checked_levels(levels).items():
            self.__find_plugin(candidate, name).permission_level = level
        self.__settings = self.__prepare(candidate.to_dict())

    def save_plugin_permission_level(self, name: str) -> None:
        """Persist one plugin's active permission level as its global default.

        Write failures propagate without changing global defaults. A plugin
        without a settings entry raises ValueError.

        Args:
            name: the plugin name, matched case-insensitively.
        """
        level = self.__find_plugin(self.__settings, name).permission_level
        candidate = self.__copy(self.__global)
        self.__find_plugin(candidate, name).permission_level = level
        self.save(candidate)

    def __default_settings(self) -> Settings:
        """Build settings from code defaults.

        Returns:
            A settings snapshot with default permission levels and no plugins.
        """
        return Settings(
            permissions=dict(DEFAULT_PERMISSION_LEVELS),
            plugins={},
            provider=None,
            model_name=None,
            thinking_effort=LITELLM_DEFAULT_THINKING_EFFORT,
            thinking_summary=LITELLM_DEFAULT_THINKING_SUMMARY,
            max_turns=AGENT_DEFAULT_MAX_TURNS,
            max_retries=AGENT_DEFAULT_MAX_RETRIES,
            turn_wait_time=AGENT_DEFAULT_TURN_WAIT_TIME,
            turn_wait_time_multiplier=AGENT_DEFAULT_TURN_WAIT_TIME_MULTIPLIER,
        )

    def __copy(self, settings: Settings) -> Settings:
        """Copy checked values into a new Settings object.

        Args:
            settings: the checked snapshot to copy.

        Returns:
            An independent object, including its permission and plugin entries.
        """
        copied = self.__default_settings()
        copied.from_dict(settings.to_dict(), self.__categories)
        return copied

    def __prepare(self, data: Mapping[str, Any]) -> Settings:
        """Strictly validate values before populating an unpublished object.

        Invalid or incomplete candidates raise ValueError.

        Args:
            data: the complete serialized candidate settings.

        Returns:
            A new checked Settings object ready for publication.
        """
        checked, _ = self.__validator.validate(data, strict=True)
        candidate = self.__default_settings()
        candidate.from_dict(checked, self.__categories)
        return candidate

    def __permission_values(
        self, permissions: Mapping[PermissionCategory, PermissionLevel]
    ) -> dict[str, str | None]:
        """Check typed permission input before converting it to plain values.

        Args:
            permissions: the category identities and typed levels supplied by a caller.

        Returns:
            Lowercase category names mapped to serialized permission levels.

        Raises:
            ValueError: when the mapping, a category or a level is invalid.
        """
        values: dict[str, str | None] = {}
        for category, level in self.__checked_levels(permissions).items():
            if category not in self.__categories:
                raise ValueError(f"Unknown permission category: {category!r}.")
            values[category.name.lower()] = level.value
        return values

    @staticmethod
    def __checked_levels(
        levels: Mapping[Any, PermissionLevel],
    ) -> Mapping[Any, PermissionLevel]:
        """Check that typed level input is a mapping of typed levels.

        Args:
            levels: the keys (categories or plugin names) and typed levels supplied
                by a caller.

        Returns:
            The same mapping, unchanged.

        Raises:
            ValueError: when the input is not a mapping or a level is not typed.
        """
        if not isinstance(levels, Mapping):
            raise ValueError(f"Permissions must be a mapping, got {levels!r}.")
        for key, level in levels.items():
            if not isinstance(level, PermissionLevel):
                raise ValueError(
                    f"Invalid permission level {level!r} for category {key!r}."
                )
        return levels

    @staticmethod
    def __find_plugin(settings: Settings, name: str) -> PluginSettings:
        """Return a plugin's settings entry by name, matched case-insensitively.

        Args:
            settings: the snapshot holding the entries.
            name: the plugin name.

        Returns:
            The entry, which callers may change in place.

        Raises:
            ValueError: when the name is not a string or has no entry.
        """
        if not isinstance(name, str) or name.lower() not in settings.plugins:
            raise ValueError(f"Unknown plugin: {name!r}.")
        return settings.plugins[name.lower()]

    def save(self, candidate: Settings) -> None:
        """Write a detached candidate before publishing it as global defaults.

        Write failures propagate without publishing the candidate.

        Args:
            candidate: the checked settings values to persist.
        """
        saved = self.__copy(candidate)
        text = yaml.safe_dump(saved.to_dict(), sort_keys=False)
        self.__storage_manager.write_file(ARANCIO_SETTINGS_FILENAME, text)
        self.__global = saved
