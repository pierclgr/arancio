"""Naming conversion utilities."""

import re


def camel_to_snake(name: str) -> str:
    """Convert a PascalCase or camelCase string to snake_case.

    Args:
        name: the PascalCase or camelCase string.

    Returns:
        The snake_case equivalent.
    """
    result = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
        "_",
        name,
    ).lower()
    return result
