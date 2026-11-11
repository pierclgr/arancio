"""Shared validation rules for settings-file fallback and explicit edits."""

from collections.abc import Callable, Collection, Mapping
from copy import deepcopy
from typing import Any, ClassVar

from arancio.core.agents.constants import AGENT_UNLIMITED_MAX_TURNS
from arancio.core.messages.base import ErrorMessage, Message, WarningMessage
from arancio.core.permissions.types import PermissionLevel
from arancio.settings.constants import (
    ARANCIO_SETTINGS_FILENAME,
    MODEL_NAME_FIELD,
    PROVIDER_FIELD,
    REASONING_EFFORT_FIELD,
    REASONING_SUMMARY_FIELD,
)
from arancio.settings.utils import locate_field


class SettingsValidator:
    """Check plain values without constructing or changing Settings objects.

    File loading returns fallback values and notices. Strict validation rejects
    invalid or incomplete candidates with ValueError. Provider and category
    metadata comes from the manager, never from a process-wide registry.

    Attributes:
        __field_rules: scalar predicates keyed by the field's dotted path in the
            file, shared by loading and explicit edits.
        __required_fields: the dotted paths that must be set for the agent to run.
        __permission_level_values: the accepted serialized permission levels.
        __defaults: copied code defaults, including default permission names.
        __sections: every section path mapped to the names of its children.
        __supported_providers: copied, lowercased provider names.
    """

    # provider membership is checked separately against the supplied metadata
    __field_rules: ClassVar[dict[str, Callable[[Any], bool]]] = {
        PROVIDER_FIELD: lambda value: value is None or isinstance(value, str),
        MODEL_NAME_FIELD: lambda value: value is None or isinstance(value, str),
        REASONING_EFFORT_FIELD: lambda value: value is None or isinstance(value, str),
        REASONING_SUMMARY_FIELD: lambda value: value is None or isinstance(value, str),
        "max_turns": lambda value: (
            (isinstance(value, str) and value == AGENT_UNLIMITED_MAX_TURNS)
            or (isinstance(value, int) and not isinstance(value, bool) and value > 0)
        ),
        "max_retries": lambda value: (
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
        ),
        "turn_wait_time": lambda value: (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= 0
        ),
        "turn_wait_time_multiplier": lambda value: (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= 0
        ),
    }

    __required_fields: ClassVar[tuple[str, ...]] = (
        PROVIDER_FIELD,
        MODEL_NAME_FIELD,
    )

    __permission_level_values: ClassVar[frozenset[str | None]] = frozenset(
        level.value for level in PermissionLevel
    )

    def __init__(
        self,
        defaults: Mapping[str, Any],
        supported_providers: Collection[str],
    ) -> None:
        """Copy code defaults and provider metadata supplied by the manager.

        Args:
            defaults: complete serialized code defaults and default permission names.
            supported_providers: the supported provider prefixes.
        """
        self.__defaults: dict[str, Any] = deepcopy(dict(defaults))
        # permissions and plugins have their own validation, so only the scalar
        # rules define the sections
        self.__sections: dict[tuple[str, ...], set[str]] = {
            (): {"permissions", "plugins"}
        }
        for field in self.__field_rules:
            keys = tuple(field.split("."))
            for depth in range(len(keys)):
                self.__sections.setdefault(keys[:depth], set()).add(keys[depth])
        self.__supported_providers: frozenset[str] = frozenset(
            provider.lower() for provider in supported_providers
        )

    def validate(
        self,
        data: Mapping[str, Any],
        *,
        strict: bool = False,
    ) -> tuple[dict[str, Any], list[Message]]:
        """Return checked values and notices, or reject an invalid strict edit.

        Missing ordinary file fields use code defaults silently. Unset provider
        and model fields produce their existing unconfigured notices. Strict
        candidates must be complete, but may retain nullable unconfigured values.

        Args:
            data: the settings mapping to check.
            strict: reject invalid, missing or unknown values instead of falling back.

        Returns:
            Checked serialized values and file-mode diagnostics. Strict validation
            returns an empty diagnostic list on success.

        Raises:
            ValueError: when data is not a mapping, a strict candidate is invalid,
                or a field has no code default.
        """
        if not isinstance(data, Mapping):
            raise ValueError(f"Settings must be a mapping, got {data!r}.")

        permissions, messages = self.__validate_permissions(data, strict=strict)
        plugins, plugin_messages = self.__validate_plugins(data, strict=strict)
        messages.extend(plugin_messages)
        normalized: dict[str, Any] = {
            "permissions": {"default": permissions},
            "plugins": plugins,
        }

        for field in self.__field_rules:
            located = locate_field(data, field)
            if strict and located is None:
                raise ValueError(f"Missing settings field: {field!r}.")
            default_located = locate_field(self.__defaults, field)
            if default_located is None:
                raise ValueError(f"No code default for settings field: {field!r}.")
            default_section, default_key = default_located
            default = default_section[default_key]
            if located is None:
                value = default
            else:
                section, key = located
                value = section[key]
            try:
                checked = self.__normalize_field(field, value)
            except ValueError:
                if strict:
                    raise
                self.__assign(normalized, field, default)
                message_cls = ErrorMessage if default is None else WarningMessage
                messages.append(
                    message_cls(
                        content=self.__invalid_field_text(field, value, default)
                    )
                )
                continue

            self.__assign(normalized, field, checked)
            if not strict and checked is None and field in self.__required_fields:
                messages.append(ErrorMessage(content=self.__not_set_text(field)))

        messages.extend(self.__check_sections(data, (), strict=strict))
        return normalized, messages

    def __check_sections(
        self,
        node: Mapping[str, Any],
        prefix: tuple[str, ...],
        *,
        strict: bool,
    ) -> list[Message]:
        """Check one section for unknown keys and for subsections that are not mappings.

        A null subsection counts as missing, so its fields use their defaults. Field
        values are checked elsewhere.

        Args:
            node: the section mapping to check.
            prefix: the path of the section; empty for the file's top level.
            strict: reject unknown keys and invalid subsections instead of warning.

        Returns:
            File-mode warnings; empty for a valid section or in strict mode.

        Raises:
            ValueError: when a strict section has an unknown key or an invalid
                subsection.
        """
        messages: list[Message] = []
        for key, child in node.items():
            child_path = (*prefix, key)
            label = ".".join(child_path)
            if key not in self.__sections[prefix]:
                if strict:
                    raise ValueError(f"Unknown settings field: {label!r}.")
                where = f' in "{".".join(prefix)}"' if prefix else ""
                messages.append(
                    WarningMessage(
                        content=(
                            f"{ARANCIO_SETTINGS_FILENAME}: unknown field {key!r}"
                            f"{where}; entry ignored."
                        )
                    )
                )
                continue
            if child_path not in self.__sections or child is None:
                continue
            if not isinstance(child, Mapping):
                text = (
                    f'{ARANCIO_SETTINGS_FILENAME}: "{label}" must be a mapping, '
                    f"got {child!r}"
                )
                if strict:
                    raise ValueError(f"{text}.")
                messages.append(WarningMessage(content=f"{text}; using defaults."))
                continue
            messages.extend(self.__check_sections(child, child_path, strict=strict))
        return messages

    @staticmethod
    def __assign(target: dict[str, Any], field: str, value: Any) -> None:
        """Store a value by its dotted path, creating the sections on the way.

        Args:
            target: the nested mapping to change.
            field: the dotted path of the value, for example ``model.name``.
            value: the value to store.
        """
        *sections, key = field.split(".")
        node = target
        for section in sections:
            node = node.setdefault(section, {})
        node[key] = value

    def __normalize_field(self, field: str, value: Any) -> Any:
        """Check one scalar field and normalize a configured provider name.

        Args:
            field: the dotted path of a known field.
            value: the candidate value.

        Returns:
            The checked value, with provider names lowercased.

        Raises:
            ValueError: when the type, range or provider name is invalid.
        """
        if not self.__field_rules[field](value):
            raise ValueError(f"Invalid value {value!r} for settings field {field!r}.")
        if field != PROVIDER_FIELD or value is None:
            return value

        provider = value.lower()
        if provider not in self.__supported_providers:
            supported = ", ".join(sorted(self.__supported_providers))
            raise ValueError(
                f"Unknown provider: {value!r}. Supported providers: {supported}."
            )
        return provider

    def __validate_permissions(
        self,
        data: Mapping[str, Any],
        *,
        strict: bool,
    ) -> tuple[dict[str, str | None], list[Message]]:
        """Check ``permissions.default`` against default names and complete it.

        Args:
            data: the complete candidate settings mapping.
            strict: require every default category and reject invalid entries.

        Returns:
            Checked lowercase category names and serialized levels, plus file-mode
            warnings. Missing or invalid file entries retain their code defaults.

        Raises:
            ValueError: when a strict permission mapping is missing or invalid.
        """
        defaults: dict[str, str | None] = self.__defaults["permissions"]["default"]
        result: dict[str, str | None] = dict(defaults)
        if "permissions" not in data:
            if strict:
                raise ValueError("Missing settings field: 'permissions'.")
            return result, []

        raw = data["permissions"]
        if not isinstance(raw, Mapping):
            text = (
                f'{ARANCIO_SETTINGS_FILENAME}: "permissions" must be a mapping, '
                f"got {raw!r}"
            )
            if strict:
                raise ValueError(f"{text}.")
            return result, [WarningMessage(content=f"{text}; using defaults.")]

        messages: list[Message] = []
        for key in raw:
            if key == "default":
                continue
            if strict:
                raise ValueError(f"Unknown permissions field: {key!r}.")
            messages.append(
                WarningMessage(
                    content=(
                        f"{ARANCIO_SETTINGS_FILENAME}: unknown field {key!r} in "
                        '"permissions"; entry ignored.'
                    )
                )
            )
        if "default" not in raw:
            if strict:
                raise ValueError("Missing permissions field: 'default'.")
            return result, messages

        raw = raw["default"]
        if not isinstance(raw, Mapping):
            text = (
                f'{ARANCIO_SETTINGS_FILENAME}: "permissions.default" must be a '
                f"mapping, got {raw!r}"
            )
            if strict:
                raise ValueError(f"{text}.")
            messages.append(WarningMessage(content=f"{text}; using defaults."))
            return result, messages

        seen: set[str] = set()
        for name, level in raw.items():
            if not isinstance(name, str) or name.lower() not in result:
                if strict:
                    raise ValueError(f"Unknown permission category: {name!r}.")
                messages.append(
                    WarningMessage(
                        content=(
                            f"{ARANCIO_SETTINGS_FILENAME}: unknown permission "
                            f"category {name!r}; entry ignored."
                        )
                    )
                )
                continue

            # check the type before membership so collections cannot leak TypeError
            if (
                not (level is None or isinstance(level, str))
                or level not in self.__permission_level_values
            ):
                if strict:
                    raise ValueError(
                        f"Invalid permission level {level!r} for category {name!r}."
                    )
                default = defaults[name.lower()]
                default_text = f'"{default}"' if default is not None else "null"
                messages.append(
                    WarningMessage(
                        content=(
                            f"{ARANCIO_SETTINGS_FILENAME}: invalid permission level "
                            f"{level!r} for category {name!r}; "
                            f"using default {default_text}."
                        )
                    )
                )
                continue

            key = name.lower()
            result[key] = level
            seen.add(key)

        if strict:
            missing = result.keys() - seen
            if missing:
                raise ValueError(
                    f"Missing permission categories: {', '.join(sorted(missing))}."
                )
        return result, messages

    def __validate_plugins(
        self,
        data: Mapping[str, Any],
        *,
        strict: bool,
    ) -> tuple[list[dict[str, dict[str, Any]]], list[Message]]:
        """Check the plugins list, keeping each valid entry under a lowercase name.

        Plugins are not looked up: an entry is valid even when its plugin does
        not exist, so it survives saving. A missing or null (YAML ``null``, ``~`` or
        empty) ``plugins`` field means no plugins, in both modes.

        Args:
            data: the complete candidate settings mapping.
            strict: reject an invalid list and invalid entries.

        Returns:
            The valid entries as one-key mappings in file order, plus file-mode
            warnings. Invalid file entries are left out.

        Raises:
            ValueError: when a strict plugins list is invalid.
        """
        raw = data.get("plugins")
        if raw is None:
            return [], []

        if not isinstance(raw, list):
            text = f'{ARANCIO_SETTINGS_FILENAME}: "plugins" must be a list, got {raw!r}'
            if strict:
                raise ValueError(f"{text}.")
            return [], [WarningMessage(content=f"{text}; no plugins loaded.")]

        result: list[dict[str, dict[str, Any]]] = []
        messages: list[Message] = []
        seen: set[str] = set()
        for item in raw:
            try:
                name, enabled, level = self.__plugin_entry(item)
                if name.lower() in seen:
                    raise ValueError(f"duplicate plugin name {name!r}")
            except ValueError as exc:
                if strict:
                    raise ValueError(f"Invalid plugins entry {item!r}: {exc}.") from exc
                messages.append(
                    WarningMessage(
                        content=(
                            f"{ARANCIO_SETTINGS_FILENAME}: invalid plugins entry "
                            f"{item!r}: {exc}; entry ignored."
                        )
                    )
                )
                continue
            seen.add(name.lower())
            result.append(
                {name.lower(): {"enabled": enabled, "permission_level": level}}
            )
        return result, messages

    @classmethod
    def __plugin_entry(cls, item: Any) -> tuple[str, bool, str | None]:
        """Check one ``{name: {enabled, permission_level}}`` plugins entry.

        Args:
            item: one element of the plugins list.

        Returns:
            The plugin name as written, its enabled flag and its serialized level.

        Raises:
            ValueError: when the entry's shape, name, flag or level is invalid.
        """
        if not isinstance(item, Mapping) or len(item) != 1:
            raise ValueError("expected one plugin name mapped to its settings")
        ((name, values),) = item.items()
        if not isinstance(name, str) or not name:
            raise ValueError(f"invalid plugin name {name!r}")
        if not isinstance(values, Mapping) or set(values) != {
            "enabled",
            "permission_level",
        }:
            raise ValueError('expected exactly "enabled" and "permission_level"')
        enabled = values["enabled"]
        level = values["permission_level"]
        if not isinstance(enabled, bool):
            raise ValueError(f'invalid "enabled" value {enabled!r}')
        # check the type before membership so collections cannot leak TypeError
        if (
            not (level is None or isinstance(level, str))
            or level not in cls.__permission_level_values
        ):
            raise ValueError(f'invalid "permission_level" value {level!r}')
        return name, enabled, level

    @staticmethod
    def __not_set_text(field: str) -> str:
        """Build the existing notice for an unconfigured provider or model.

        Args:
            field: the dotted path of the unset field.

        Returns:
            The notice text naming the settings file and field.
        """
        return f'{ARANCIO_SETTINGS_FILENAME}: "{field}" is not set.'

    @staticmethod
    def __invalid_field_text(field: str, value: Any, default: Any) -> str:
        """Build the existing notice for an invalid file field and its fallback.

        Args:
            field: the dotted path of the invalid field.
            value: the rejected value.
            default: the code-default fallback.

        Returns:
            The notice text naming the file, field, value and fallback.
        """
        return (
            f'{ARANCIO_SETTINGS_FILENAME}: invalid value {value!r} for "{field}"; '
            f"using default {default!r}."
        )
