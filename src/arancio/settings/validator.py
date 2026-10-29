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
        __defaults: copied code defaults, including known permission names.
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
            defaults: complete serialized code defaults and permission names.
            supported_providers: the supported provider prefixes.
        """
        self.__defaults: dict[str, Any] = dict(defaults)
        self.__defaults["permissions"] = dict(defaults["permissions"])
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
        normalized: dict[str, Any] = {"permissions": permissions}

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

        known_fields = {"permissions", *self.__SETTINGS_FIELD_RULES}
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
        """Check permissions against supplied names and return a complete mapping.

        Args:
            data: the complete candidate settings mapping.
            strict: require every known category and reject invalid entries.

        Returns:
            Checked lowercase category names and serialized levels, plus file-mode
            warnings. Missing or invalid file entries retain their code defaults.

        Raises:
            ValueError: when a strict permission mapping is missing or invalid.
        """
        result: dict[str, str | None] = dict(self.__defaults["permissions"])
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
                default = self.__defaults["permissions"][name.lower()]
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
