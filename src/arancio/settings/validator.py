"""Shared validation rules for settings-file fallback and explicit edits."""

from collections.abc import Callable, Collection, Mapping
from typing import Any, ClassVar

from arancio.core.agents.constants import AGENT_UNLIMITED_MAX_TURNS
from arancio.core.messages.base import ErrorMessage, Message, WarningMessage
from arancio.core.permissions.types import PermissionLevel
from arancio.settings.constants import ARANCIO_SETTINGS_FILENAME


class SettingsValidator:
    """Check plain values without constructing or changing Settings objects.

    File loading returns fallback values and notices. Strict validation rejects
    invalid or incomplete candidates with ValueError. Provider and category
    metadata comes from the manager, never from a process-wide registry.

    Attributes:
        __SETTINGS_FIELD_RULES: scalar predicates shared by loading and explicit edits.
        __PERMISSION_LEVEL_VALUES: the accepted serialized permission levels.
        __defaults: copied code defaults, including default permission names.
        __supported_providers: copied, lowercased provider names.
    """

    # provider membership is checked separately against the supplied metadata
    __SETTINGS_FIELD_RULES: ClassVar[dict[str, Callable[[Any], bool]]] = {
        "provider": lambda value: value is None or isinstance(value, str),
        "model_name": lambda value: value is None or isinstance(value, str),
        "thinking_effort": lambda value: value is None or isinstance(value, str),
        "thinking_summary": lambda value: value is None or isinstance(value, str),
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

    __PERMISSION_LEVEL_VALUES: ClassVar[frozenset[str | None]] = frozenset(
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
        self.__defaults: dict[str, Any] = dict(defaults)
        self.__defaults["permissions"] = {
            "default": dict(defaults["permissions"]["default"])
        }
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
            ValueError: when data is not a mapping or a strict candidate is invalid.
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

        for field in self.__SETTINGS_FIELD_RULES:
            if strict and field not in data:
                raise ValueError(f"Missing settings field: {field!r}.")
            default = self.__defaults[field]
            value = data.get(field, default)
            try:
                normalized[field] = self.__normalize_field(field, value)
            except ValueError:
                if strict:
                    raise
                normalized[field] = default
                message_cls = ErrorMessage if default is None else WarningMessage
                messages.append(
                    message_cls(
                        content=self.__invalid_field_text(field, value, default)
                    )
                )
                continue

            if (
                not strict
                and field in ("provider", "model_name")
                and normalized[field] is None
            ):
                messages.append(ErrorMessage(content=self.__not_set_text(field)))

        known_fields = {"permissions", "plugins", *self.__SETTINGS_FIELD_RULES}
        for key in data:
            if key in known_fields:
                continue
            if strict:
                raise ValueError(f"Unknown settings field: {key!r}.")
            messages.append(
                WarningMessage(
                    content=(
                        f"{ARANCIO_SETTINGS_FILENAME}: unknown field {key!r}; "
                        "entry ignored."
                    )
                )
            )

        return normalized, messages

    def __normalize_field(self, field: str, value: Any) -> Any:
        """Check one scalar field and normalize a configured provider name.

        Args:
            field: a known settings field name.
            value: the candidate value.

        Returns:
            The checked value, with provider names lowercased.

        Raises:
            ValueError: when the type, range or provider name is invalid.
        """
        if not self.__SETTINGS_FIELD_RULES[field](value):
            raise ValueError(f"Invalid value {value!r} for settings field {field!r}.")
        if field != "provider" or value is None:
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
                or level not in self.__PERMISSION_LEVEL_VALUES
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
        not exist, so it survives saving.

        Args:
            data: the complete candidate settings mapping.
            strict: reject a missing list and invalid entries.

        Returns:
            The valid entries as one-key mappings in file order, plus file-mode
            warnings. Invalid file entries are left out.

        Raises:
            ValueError: when a strict plugins list is missing or invalid.
        """
        if "plugins" not in data:
            if strict:
                raise ValueError("Missing settings field: 'plugins'.")
            return [], []

        raw = data["plugins"]
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
            or level not in cls.__PERMISSION_LEVEL_VALUES
        ):
            raise ValueError(f'invalid "permission_level" value {level!r}')
        return name, enabled, level

    @staticmethod
    def __not_set_text(field: str) -> str:
        """Build the existing notice for an unconfigured provider or model.

        Args:
            field: the unset field name.

        Returns:
            The notice text naming the settings file and field.
        """
        return f'{ARANCIO_SETTINGS_FILENAME}: "{field}" is not set.'

    @staticmethod
    def __invalid_field_text(field: str, value: Any, default: Any) -> str:
        """Build the existing notice for an invalid file field and its fallback.

        Args:
            field: the invalid field name.
            value: the rejected value.
            default: the code-default fallback.

        Returns:
            The notice text naming the file, field, value and fallback.
        """
        return (
            f'{ARANCIO_SETTINGS_FILENAME}: invalid value {value!r} for "{field}"; '
            f"using default {default!r}."
        )
