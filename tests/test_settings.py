"""Tests for the Settings data holder and its YAML-serializable form."""

from arancio.core.constants.agent import (
    AGENT_DEFAULT_MAX_RETRIES,
    AGENT_DEFAULT_MAX_TURNS,
    AGENT_DEFAULT_TURN_WAIT_TIME,
    AGENT_DEFAULT_TURN_WAIT_TIME_MULTIPLIER,
)
from arancio.core.constants.litellm import (
    LITELLM_DEFAULT_THINKING_EFFORT,
    LITELLM_DEFAULT_THINKING_SUMMARY,
)
from arancio.core.types.permissions import PermissionCategory, PermissionLevel
from arancio.settings.settings import Settings


def test_default_values():
    """The default settings match the registered code defaults."""
    settings = Settings.default()

    assert settings.permissions == {
        category: PermissionLevel.ASK for category in PermissionCategory
    }
    assert settings.model_id is None
    assert settings.summary_model_id is None
    assert settings.thinking_effort == LITELLM_DEFAULT_THINKING_EFFORT
    assert settings.thinking_summary == LITELLM_DEFAULT_THINKING_SUMMARY
    assert settings.max_turns == AGENT_DEFAULT_MAX_TURNS
    assert settings.max_retries == AGENT_DEFAULT_MAX_RETRIES
    assert settings.turn_wait_time == AGENT_DEFAULT_TURN_WAIT_TIME
    assert settings.turn_wait_time_multiplier == AGENT_DEFAULT_TURN_WAIT_TIME_MULTIPLIER


def test_to_dict_uses_plain_types():
    """``to_dict`` flattens permissions to category-name to level-value strings."""
    data = Settings.default().to_dict()

    assert data["permissions"] == {
        "READ": "ask",
        "WRITE": "ask",
        "WEB": "ask",
        "EXECUTE": "ask",
    }
    assert data["model_id"] is None
    assert data["max_turns"] == AGENT_DEFAULT_MAX_TURNS
    assert data["max_retries"] == AGENT_DEFAULT_MAX_RETRIES


def test_to_from_dict_roundtrip():
    """``from_dict`` reverses ``to_dict``, restoring enum keys and values."""
    settings = Settings(
        permissions={
            PermissionCategory.READ: PermissionLevel.AUTO,
            PermissionCategory.EXECUTE: PermissionLevel.ASK,
        },
        model_id="openai/gpt-4o",
        summary_model_id="openai/gpt-4o-mini",
        thinking_effort="high",
        thinking_summary=None,
        max_turns=7,
        max_retries=4,
        turn_wait_time=1.5,
        turn_wait_time_multiplier=3.0,
    )

    assert Settings.from_dict(settings.to_dict()) == settings
