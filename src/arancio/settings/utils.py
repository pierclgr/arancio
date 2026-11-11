"""Helpers shared by the settings manager and validator."""

from typing import Any


def locate_field(data: Any, field: Any) -> tuple[dict[str, Any], str] | None:
    """Find the section that holds a dotted field path, and the field's key in it.

    Args:
        data: the nested settings data to search.
        field: the dotted path of a field or section, for example ``model.name``.

    Returns:
        The section and the last key of the path, or ``None`` when the path is not
        a string, a section on the way is missing or not a mapping, or the key is
        not in its section.
    """
    if not isinstance(field, str):
        return None
    *parents, key = field.split(".")
    section = data
    for parent in parents:
        if not isinstance(section, dict) or parent not in section:
            return None
        section = section[parent]
    if not isinstance(section, dict) or key not in section:
        return None
    return section, key
