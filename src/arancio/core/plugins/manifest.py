"""Parsing and validation of a plugin's ``manifest.yml`` metadata."""

from dataclasses import dataclass
from typing import Any, Callable, ClassVar

from arancio.core.messages import Message, WarningMessage


@dataclass(frozen=True)
class PluginManifest:
    """Metadata a plugin declares in its ``manifest.yml``.

    Attributes:
        name: display name, defaulting to the plugin's folder name.
        version: the plugin's own version string.
        description: one line describing what the plugin does.
        author: who wrote the plugin.
        enabled: whether the plugin is loaded at all.
    """

    name: str
    version: str = "0.0.0"
    description: str = ""
    author: str = ""
    enabled: bool = True


class PluginManifestValidator:
    """Validate a parsed ``manifest.yml`` mapping into a :class:`PluginManifest`.

    Follows the same policy as
    :class:`~arancio.settings.validator.SettingsValidator`: **nothing raises**,
    a missing field silently takes its default, an invalid field falls back to
    its default and is reported as a
    :class:`~arancio.core.messages.WarningMessage`, and a top-level key naming
    none of the known fields is dropped with a warning. Every field here has a
    non-``None`` default, so the manifest never produces an error — a plugin
    fails to load because of its folder or its module, not its metadata.

    The caller reads and YAML-parses the file and handles a missing or
    unparseable one; this validator only ever sees an already-parsed mapping.
    """

    # each predicate composes the unit checks with the and/or the field needs
    _FIELD_RULES: ClassVar[tuple[tuple[str, Any, Callable[[Any], bool]], ...]] = (
        ("version", "0.0.0", lambda v: isinstance(v, str)),
        ("description", "", lambda v: isinstance(v, str)),
        ("author", "", lambda v: isinstance(v, str)),
        ("enabled", True, lambda v: isinstance(v, bool)),
    )

    @classmethod
    def validate(
        cls, data: dict[str, Any], directory_name: str, location: str
    ) -> tuple[PluginManifest, list[Message]]:
        """Validate a parsed manifest mapping into a plugin manifest.

        ``name`` is handled apart from the other fields because its default is
        not a constant: a plugin that does not name itself is named after its
        folder, which is its identity anyway.

        Args:
            data: the mapping parsed from the manifest file.
            directory_name: the plugin's folder name, used as the default
                display name.
            location: the manifest's path as it should read in messages, e.g.
                ``plugins/tool_call_logger/manifest.yml``.

        Returns:
            A ``(manifest, messages)`` pair: the validated manifest, and the
            messages describing any fallback applied.
        """
        messages: list[Message] = []
        normalized: dict[str, Any] = {}

        rules = (
            ("name", directory_name, lambda v: isinstance(v, str)),
            *cls._FIELD_RULES,
        )
        known_fields = {field for field, _, _ in rules}
        for field, default, is_valid in rules:
            if field not in data:
                normalized[field] = default
                continue
            value = data[field]
            if is_valid(value):
                normalized[field] = value
                continue
            normalized[field] = default
            messages.append(
                WarningMessage(
                    content=cls._invalid_field_text(location, field, value, default)
                )
            )

        for key in data:
            if key not in known_fields:
                messages.append(
                    WarningMessage(
                        content=f"{location}: unknown field {key!r}; entry ignored."
                    )
                )

        return PluginManifest(**normalized), messages

    @staticmethod
    def _invalid_field_text(location: str, field: str, value: Any, default: Any) -> str:
        """Build the message text for an invalid manifest field.

        Args:
            location: the manifest's path as it should read in messages.
            field: the name of the invalid field.
            value: the offending value read from the manifest.
            default: the default value the field falls back to.

        Returns:
            The message text naming the file, field, offending value and
            default.
        """
        return (
            f"{location}: invalid value {value!r} for {field!r}; "
            f"using default {default!r}."
        )
