"""Tests for the generic single-purpose validation predicates."""

from arancio.settings.utils.validations import (
    is_float,
    is_int,
    is_known_provider,
    is_negative,
    is_none,
    is_number,
    is_positive,
    is_str,
)


def test_is_none():
    """``is_none`` matches only ``None``."""
    assert is_none(None) is True
    assert is_none(0) is False
    assert is_none("") is False


def test_is_str():
    """``is_str`` matches only strings."""
    assert is_str("hi") is True
    assert is_str(1) is False
    assert is_str(None) is False


def test_is_int_excludes_bool():
    """``is_int`` matches integers but rejects booleans."""
    assert is_int(1) is True
    assert is_int(-1) is True
    assert is_int(True) is False
    assert is_int(False) is False
    assert is_int(1.0) is False


def test_is_float():
    """``is_float`` matches only floats."""
    assert is_float(1.0) is True
    assert is_float(1) is False


def test_is_number_accepts_int_and_float_but_not_bool():
    """``is_number`` accepts int or float, still rejecting booleans."""
    assert is_number(1) is True
    assert is_number(1.5) is True
    assert is_number(True) is False
    assert is_number("1") is False


def test_is_positive():
    """``is_positive`` matches strictly-positive values only."""
    assert is_positive(1) is True
    assert is_positive(0) is False
    assert is_positive(-1) is False


def test_is_negative():
    """``is_negative`` matches strictly-negative values only."""
    assert is_negative(-1) is True
    assert is_negative(0) is False
    assert is_negative(1) is False


def test_is_known_provider_is_case_insensitive():
    """``is_known_provider`` matches a recognized provider name in any case."""
    assert is_known_provider("openai") is True
    assert is_known_provider("OpenAI") is True
    assert is_known_provider("notreal") is False
