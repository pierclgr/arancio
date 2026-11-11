"""Settings values, validation and YAML persistence through generic Storage."""

from collections.abc import Collection, Mapping
from typing import Any

import yaml

from arancio.core.agents.constants import DEFAULT_TURN
from arancio.core.clients.constants import (
    LITELLM_DEFAULT_REASONING_EFFORT,
    LITELLM_DEFAULT_REASONING_SUMMARY,
)
from arancio.core.clients.types import Model, Reasoning
from arancio.core.messages.base import ErrorMessage, Message, WarningMessage
from arancio.core.permissions.constants import DEFAULT_PERMISSION_LEVELS
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.settings.base import PluginSettings, Settings
from arancio.settings.constants import (
    ARANCIO_SETTINGS_FILENAME,
    MODEL_NAME_FIELD,
    PERMISSIONS_FIELD,
    REASONING_EFFORT_FIELD,
)
from arancio.settings.utils import locate_field
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
        __settings_validator: shared validation for file loading and explicit edits.
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
        self.__settings_validator = SettingsValidator(
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

        checked, field_messages = self.__settings_validator.validate(data)
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

        Permissions are merged rather than replaced. A model name requires a
        configured provider, and changing reasoning effort requires a configured model.

        Args:
            field: the dotted path of the setting to change, as in the settings
                file (for example ``model.name`` or ``turn.max``).
            value: the new value, or typed category-to-level changes for permissions.

        Raises:
            ValueError: when the field or value is invalid, or a model is required.
        """
        data = self.__settings.to_dict()
        located = locate_field(data, field)
        if located is None:
            raise ValueError(f"Unknown editable settings field: {field!r}.")
        if field == REASONING_EFFORT_FIELD:
            self.__settings.model.check_configured()

        if field == PERMISSIONS_FIELD:
            data[field]["default"].update(self.__permission_values(value))
        else:
            section, key = located
            section[key] = value

        candidate = self.__prepare(data)
        if field == MODEL_NAME_FIELD:
            candidate.model.check_configured()
        self.__settings = candidate

    def restore_active(
        self,
        *,
        model: Model,
        permissions: Mapping[PermissionCategory, PermissionLevel],
    ) -> None:
        """Restore the session's model and permissions on top of global defaults.

        Plugins and loop limits retain their global-default values. This operation
        does not save settings or receive a Sessions object. Invalid values or
        incomplete permissions raise ValueError.

        Args:
            model: the restored model section, reasoning options included.
            permissions: the complete restored permission map for default categories.
        """
        values = self.__permission_values(permissions)
        data = self.__global.to_dict()
        data.update(
            model=model.to_dict(),
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
        self, field: str, permission_category: PermissionCategory | None = None
    ) -> None:
        """Persist one active field or permission category as a global default.

        Write failures propagate without changing global defaults.

        Args:
            field: the dotted path of the settings field whose active value to save.
            permission_category: the known category to save when field is permissions.

        Raises:
            ValueError: when the field or category is invalid.
        """
        active = locate_field(self.__settings.to_dict(), field)
        if active is None:
            raise ValueError(f"Unknown settings field: {field!r}.")
        if field == "permissions" and permission_category not in self.__categories:
            raise ValueError(f"Unknown permission category: {permission_category!r}.")
        if field != "permissions" and permission_category is not None:
            raise ValueError(
                f"Category is only valid for permissions: {permission_category!r}."
            )

        if field == "permissions":
            candidate = self.__copy(self.__global)
            candidate.permissions[permission_category] = self.__settings.permissions[
                permission_category
            ]
        else:
            data = self.__global.to_dict()
            section, key = locate_field(data, field)
            section[key] = active[0][key]
            candidate = self.__prepare(data)
        self.save(candidate)

    def set_plugin_permission_levels(
        self, permission_levels: Mapping[str, PermissionLevel]
    ) -> None:
        """Validate and update plugin permission levels without saving them.

        An invalid mapping or level, or a plugin without a settings entry, raises
        ValueError and changes nothing.

        Args:
            permission_levels: plugin names, matched case-insensitively, mapped to
                typed levels.
        """
        candidate = self.__copy(self.__settings)
        for name, permission_level in self.__checked_levels(permission_levels).items():
            self.__find_plugin(candidate, name).permission_level = permission_level
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
            A settings snapshot with default permission levels, no plugins and an
            unconfigured model.
        """
        return Settings(
            permissions=dict(DEFAULT_PERMISSION_LEVELS),
            plugins={},
            model=Model(
                provider=None,
                name=None,
                reasoning=Reasoning(
                    effort=LITELLM_DEFAULT_REASONING_EFFORT,
                    summary=LITELLM_DEFAULT_REASONING_SUMMARY,
                ),
            ),
            turn=DEFAULT_TURN,
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
        checked, _ = self.__settings_validator.validate(data, strict=True)
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
        permission_levels: Mapping[Any, PermissionLevel],
    ) -> Mapping[Any, PermissionLevel]:
        """Check that typed level input is a mapping of typed levels.

        Args:
            permission_levels: the keys (categories or plugin names) and typed
                levels supplied by a caller.

        Returns:
            The same mapping, unchanged.

        Raises:
            ValueError: when the input is not a mapping or a level is not typed.
        """
        if not isinstance(permission_levels, Mapping):
            raise ValueError(
                f"Permissions must be a mapping, got {permission_levels!r}."
            )
        for key, permission_level in permission_levels.items():
            if not isinstance(permission_level, PermissionLevel):
                raise ValueError(
                    f"Invalid permission level {permission_level!r} "
                    f"for category {key!r}."
                )
        return permission_levels

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
