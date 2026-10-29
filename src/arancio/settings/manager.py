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
from arancio.settings.constants import ARANCIO_SETTINGS_FILENAME
from arancio.settings.settings import Settings
from arancio.settings.validator import SettingsValidator
from arancio.storage.manager import StorageManager


class SettingsManager:
    """Own active settings and global defaults without applying them to Core.

    Only this facade constructs and copies Settings objects. Callers edit active
    values through its methods and read detached snapshots through properties.
    File policy belongs here; Storage supplies UTF-8 reads and writes.

    Attributes:
        __storage_manager: the injected generic file I/O service.
        __categories: category identities registered before construction.
        __validator: shared validation for file loading and explicit edits.
        __settings: the active conversation values.
        __global_settings: the defaults selected by load or a successful save.
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
        self.__categories: tuple[PermissionCategory, ...] = tuple(PermissionCategory)
        self.__global_settings: Settings = self.__default_settings()
        self.__settings: Settings = self.__copy_settings(self.__global_settings)
        self.__validator = SettingsValidator(
            self.__global_settings.to_dict(), supported_providers
        )

    @property
    def settings(self) -> Settings:
        """Return a detached copy of the active settings.

        Returns:
            A snapshot whose fields and permission dictionary can be changed
            without changing manager-owned values.
        """
        return self.__copy_settings(self.__settings)

    @property
    def global_settings(self) -> Settings:
        """Return a detached copy of the loaded or successfully saved defaults.

        Returns:
            An independent snapshot of the global-default values.
        """
        return self.__copy_settings(self.__global_settings)

    def load_settings(self) -> tuple[Settings, list[Message]]:
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
        active = self.__copy_settings(defaults)
        self.__global_settings = defaults
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
            value = self.__permission_values(value)
        if field == "permissions":
            data[field].update(value)
        else:
            data[field] = value

        candidate = self.__prepare_settings(data)
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
            permissions: the complete restored permission map for known categories.
        """
        values = self.__permission_values(permissions)
        data = self.__global_settings.to_dict()
        data.update(
            provider=provider,
            model_name=model_name,
            thinking_effort=thinking_effort,
            permissions=values,
        )
        self.__settings = self.__prepare_settings(data)

    def reset_to_global_settings(self) -> None:
        """Replace active values with global defaults without file I/O."""
        self.__settings = self.__copy_settings(self.__global_settings)

    def reset_to_default_settings(self) -> None:
        """Write code defaults, then replace both active and global settings.

        Write failures propagate and leave both in-memory snapshots unchanged.
        """
        candidate = self.__default_settings()
        self.save_settings(candidate)
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

        candidate = self.__copy_settings(self.__global_settings)
        if field == "permissions":
            candidate.permissions[category] = self.__settings.permissions[category]
        else:
            setattr(candidate, field, getattr(self.__settings, field))
        self.save_settings(candidate)

    def __default_settings(self) -> Settings:
        """Build settings from code defaults and the known categories.

        Returns:
            A settings snapshot with built-in defaults and additional categories at ASK.
        """
        return Settings(
            permissions={
                category: DEFAULT_PERMISSION_LEVELS.get(category, PermissionLevel.ASK)
                for category in self.__categories
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

    def __copy_settings(self, settings: Settings) -> Settings:
        """Copy checked values into a new Settings object.

        Args:
            settings: the checked snapshot to copy.

        Returns:
            An independent object, including its permission dictionary.
        """
        copied = self.__default_settings()
        copied.from_dict(settings.to_dict(), self.__categories)
        return copied

    def __prepare_settings(self, data: Mapping[str, Any]) -> Settings:
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
        if not isinstance(permissions, Mapping):
            raise ValueError(f"Permissions must be a mapping, got {permissions!r}.")
        values: dict[str, str | None] = {}
        for category, level in permissions.items():
            if category not in self.__categories:
                raise ValueError(f"Unknown permission category: {category!r}.")
            if not isinstance(level, PermissionLevel):
                raise ValueError(
                    f"Invalid permission level {level!r} "
                    f"for category {category.name!r}."
                )
            values[category.name.lower()] = level.value
        return values

    def save_settings(self, candidate: Settings) -> None:
        """Write a detached candidate before publishing it as global defaults.

        Write failures propagate without publishing the candidate.

        Args:
            candidate: the checked settings values to persist.
        """
        saved = self.__copy_settings(candidate)
        text = yaml.safe_dump(saved.to_dict(), sort_keys=False)
        self.__storage_manager.write_file(ARANCIO_SETTINGS_FILENAME, text)
        self.__global_settings = saved
