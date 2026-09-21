"""Validation of the parsed settings mapping into a usable settings snapshot."""

from typing import Any, Callable, ClassVar

from arancio.core.constants.agent import (
    AGENT_DEFAULT_MAX_RETRIES,
    AGENT_DEFAULT_MAX_TURNS,
    AGENT_DEFAULT_TURN_WAIT_TIME,
    AGENT_DEFAULT_TURN_WAIT_TIME_MULTIPLIER,
    AGENT_UNLIMITED_MAX_TURNS,
)
from arancio.core.constants.litellm import (
    LITELLM_DEFAULT_THINKING_EFFORT,
    LITELLM_DEFAULT_THINKING_SUMMARY,
    LITELLM_PROVIDER_NAMES,
)
from arancio.core.messages import ErrorMessage, Message, WarningMessage
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.settings.settings import Settings


class SettingsValidator:
    """Validate a parsed ``settings.yml`` mapping into a usable settings snapshot.

    Every field falls back to its code default when missing or invalid, so a malformed
    settings file never prevents the app from starting. A missing field is silently
    defaulted; an invalid field falls back to its default and is reported through a
    :class:`~arancio.core.messages.WarningMessage`, or an
    :class:`~arancio.core.messages.ErrorMessage` when the field's default is ``None``.
    The caller (:class:`~arancio.storage.manager.StorageManager`) is responsible for
    reading and YAML-parsing the file and for handling a missing or unparseable file;
    this validator only ever sees an already-parsed mapping.
    """

    # each predicate composes the unit checks with the and/or the field needs
    _FIELD_RULES: ClassVar[tuple[tuple[str, Any, Callable[[Any], bool]], ...]] = (
        (
            "provider",
            None,
            lambda v: isinstance(v, str) and v.lower() in LITELLM_PROVIDER_NAMES,
        ),
        ("model_name", None, lambda v: isinstance(v, str)),
        (
            "thinking_effort",
            LITELLM_DEFAULT_THINKING_EFFORT,
            lambda v: v is None or isinstance(v, str),
        ),
        (
            "thinking_summary",
            LITELLM_DEFAULT_THINKING_SUMMARY,
            lambda v: v is None or isinstance(v, str),
        ),
        (
            "max_turns",
            AGENT_DEFAULT_MAX_TURNS,
            lambda v: (
                v == AGENT_UNLIMITED_MAX_TURNS
                or (isinstance(v, int) and not isinstance(v, bool) and v > 0)
            ),
        ),
        (
            "max_retries",
            AGENT_DEFAULT_MAX_RETRIES,
            lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0,
        ),
        (
            "turn_wait_time",
            AGENT_DEFAULT_TURN_WAIT_TIME,
            lambda v: (
                isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
            ),
        ),
        (
            "turn_wait_time_multiplier",
            AGENT_DEFAULT_TURN_WAIT_TIME_MULTIPLIER,
            lambda v: (
                isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
            ),
        ),
    )

    @classmethod
    def validate(cls, data: dict[str, Any]) -> tuple[Settings, list[Message]]:
        """Validate a parsed settings mapping into a settings snapshot.

        Every field is checked individually: a missing field silently falls
        back to its default, and an invalid field falls back to its default
        and is reported as a warning, or an error when the field's default is
        ``None``. A top-level key that names none of the known fields is
        reported as a warning and ignored, the same treatment as an unknown
        permission category. ``provider``/``model_name`` are the exception to
        the "missing is silent" rule: since their default is ``None``, a
        missing key resolves to the same unconfigured ``None`` as a present
        ``null`` value, so both are reported the same way.

        Args:
            data: the mapping parsed from the settings file (the caller
                handles a missing or unparseable file before reaching here).

        Returns:
            A ``(settings, messages)`` pair: the validated settings, and the
            messages to surface in the UI describing any fallback applied.
        """
        permissions, messages = cls._validate_permissions(data)
        normalized: dict[str, Any] = {"permissions": permissions}

        known_fields = {"permissions", *(field for field, _, _ in cls._FIELD_RULES)}
        for field, default, is_valid in cls._FIELD_RULES:
            if field not in data and default is not None:
                normalized[field] = default
                continue
            value = data.get(field)
            if is_valid(value):
                normalized[field] = value
                continue
            normalized[field] = default
            message_cls = ErrorMessage if default is None else WarningMessage
            text = (
                cls._not_set_text(field)
                if value is None and default is None
                else cls._invalid_field_text(field, value, default)
            )
            messages.append(message_cls(content=text))

        for key in data:
            if key not in known_fields:
                messages.append(
                    WarningMessage(
                        content=f"settings.yml: unknown field {key!r}; entry ignored."
                    )
                )

        return Settings.from_dict(normalized), messages

    @classmethod
    def _validate_permissions(
        cls, data: dict[str, Any]
    ) -> tuple[dict[str, str | None], list[Message]]:
        """Validate the ``permissions`` field, always returning a complete mapping.

        A category missing from the file is silently defaulted to
        ``"ask"``, matching the missing-field rule applied to every other
        field. A known category with an invalid level falls back to
        ``"ask"`` with a warning; an unknown category name is dropped with a
        warning since there is no category to default it to.

        Args:
            data: the raw mapping parsed from the settings file.

        Returns:
            A ``(permissions, messages)`` pair: the permissions as a flat
            mapping of lowercase category name to level value (covering
            every known category), and the warnings produced by invalid
            entries.
        """
        result: dict[str, str | None] = {
            category.name.lower(): PermissionLevel.ASK.value
            for category in PermissionCategory
        }
        if "permissions" not in data:
            return result, []

        raw = data["permissions"]
        if not isinstance(raw, dict):
            return result, [
                WarningMessage(
                    content=(
                        f'settings.yml: "permissions" must be a mapping, got {raw!r}; '
                        'using defaults (all categories "ask").'
                    )
                )
            ]

        valid_levels = {level.value for level in PermissionLevel}
        messages: list[Message] = []
        for name, level in raw.items():
            if (
                not isinstance(name, str)
                or name.upper() not in PermissionCategory.__members__
            ):
                messages.append(
                    WarningMessage(
                        content=(
                            f"settings.yml: unknown permission category {name!r}; "
                            "entry ignored."
                        )
                    )
                )
                continue
            if level not in valid_levels:
                messages.append(
                    WarningMessage(
                        content=(
                            f"settings.yml: invalid permission level {level!r} for "
                            f'category {name!r}; using default "ask".'
                        )
                    )
                )
                continue
            result[name.lower()] = level

        return result, messages

    @staticmethod
    def _not_set_text(field: str) -> str:
        """Build the message text for a null-default field left unset.

        Args:
            field: the name of the unset field.

        Returns:
            The message text naming the file and the field.
        """
        return f'settings.yml: "{field}" is not set.'

    @staticmethod
    def _invalid_field_text(field: str, value: Any, default: Any) -> str:
        """Build the message text for an invalid top-level field.

        Args:
            field: the name of the invalid field.
            value: the offending value read from the settings file.
            default: the default value the field falls back to.

        Returns:
            The message text naming the file, field, offending value and
            default.
        """
        return (
            f'settings.yml: invalid value {value!r} for "{field}"; '
            f"using default {default!r}."
        )
