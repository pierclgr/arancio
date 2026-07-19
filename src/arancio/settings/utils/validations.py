"""Generic single-purpose value validation predicates."""

from typing import Any

from arancio.core.constants.litellm import LITELLM_PROVIDER_NAMES


def is_none(value: Any) -> bool:
    """Return whether a value is ``None``.

    Args:
        value: the value to check.

    Returns:
        ``True`` when ``value`` is ``None``.
    """
    return value is None


def is_str(value: Any) -> bool:
    """Return whether a value is a string.

    Args:
        value: the value to check.

    Returns:
        ``True`` when ``value`` is a ``str``.
    """
    return isinstance(value, str)


def is_int(value: Any) -> bool:
    """Return whether a value is an integer, excluding booleans.

    Args:
        value: the value to check.

    Returns:
        ``True`` when ``value`` is an ``int`` and not a ``bool``.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def is_float(value: Any) -> bool:
    """Return whether a value is a float.

    Args:
        value: the value to check.

    Returns:
        ``True`` when ``value`` is a ``float``.
    """
    return isinstance(value, float)


def is_number(value: Any) -> bool:
    """Return whether a value is an integer or a float.

    Args:
        value: the value to check.

    Returns:
        ``True`` when ``value`` satisfies :func:`is_int` or :func:`is_float`.
    """
    return is_int(value) or is_float(value)


def is_positive(value: Any) -> bool:
    """Return whether a numeric value is strictly greater than zero.

    Args:
        value: the numeric value to check.

    Returns:
        ``True`` when ``value > 0``.
    """
    return value > 0


def is_negative(value: Any) -> bool:
    """Return whether a numeric value is strictly less than zero.

    Args:
        value: the numeric value to check.

    Returns:
        ``True`` when ``value < 0``.
    """
    return value < 0


def is_known_provider(value: str) -> bool:
    """Return whether a string names a LiteLLM-supported provider.

    Args:
        value: the provider name to check, in any case.

    Returns:
        ``True`` when ``value`` lowercased is in
        :data:`~arancio.core.constants.litellm.LITELLM_PROVIDER_NAMES`.
    """
    return value.lower() in LITELLM_PROVIDER_NAMES
