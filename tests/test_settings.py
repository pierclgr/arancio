"""Tests for the Settings data holder and its YAML-serializable form."""

import pytest

from arancio.core.constants.agent import (
    AGENT_DEFAULT_MAX_RETRIES,
    AGENT_DEFAULT_MAX_TURNS,
    AGENT_DEFAULT_TURN_WAIT_TIME,
    AGENT_DEFAULT_TURN_WAIT_TIME_MULTIPLIER,
)
from arancio.core.constants.litellm import (
    LITELLM_DEFAULT_THINKING_EFFORT,
    LITELLM_DEFAULT_THINKING_SUMMARY,
    LITELLM_PROVIDER_NAMES,
)
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.settings.settings import Settings


def test_default_values():
    """The default settings match the registered code defaults."""
    settings = Settings.default()

    assert settings.permissions == {
        category: PermissionLevel.ASK for category in PermissionCategory
    }
    assert settings.provider is None
    assert settings.model_name is None
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
        "read": "ask",
        "write": "ask",
        "web": "ask",
        "execute": "ask",
    }
    assert data["provider"] is None
    assert data["model_name"] is None
    assert data["max_turns"] == AGENT_DEFAULT_MAX_TURNS
    assert data["max_retries"] == AGENT_DEFAULT_MAX_RETRIES


def test_to_from_dict_roundtrip():
    """``from_dict`` reverses ``to_dict``, restoring enum keys and values."""
    settings = Settings(
        permissions={
            PermissionCategory.READ: PermissionLevel.AUTO,
            PermissionCategory.WRITE: PermissionLevel.NONE,
            PermissionCategory.WEB: PermissionLevel.NONE,
            PermissionCategory.EXECUTE: PermissionLevel.ASK,
        },
        provider="openai",
        model_name="gpt-4o",
        thinking_effort="high",
        thinking_summary=None,
        max_turns=7,
        max_retries=4,
        turn_wait_time=1.5,
        turn_wait_time_multiplier=3.0,
    )

    assert Settings.from_dict(settings.to_dict()) == settings


def test_to_dict_renders_none_as_null():
    """``to_dict`` renders a NONE level as plain ``None`` (YAML null)."""
    settings = Settings(
        permissions={
            PermissionCategory.READ: PermissionLevel.AUTO,
            PermissionCategory.WRITE: PermissionLevel.NONE,
            PermissionCategory.WEB: PermissionLevel.NONE,
            PermissionCategory.EXECUTE: PermissionLevel.NONE,
        },
        provider=None,
        model_name=None,
        thinking_effort=None,
        thinking_summary=None,
        max_turns=1,
        max_retries=1,
        turn_wait_time=1.0,
        turn_wait_time_multiplier=1.0,
    )

    assert settings.to_dict()["permissions"] == {
        "read": "auto",
        "write": None,
        "web": None,
        "execute": None,
    }


def test_model_id_joins_provider_and_model_name():
    """``model_id`` joins the provider and model name fields."""
    settings = Settings.default()
    settings.provider = "openai"
    settings.model_name = "gpt-4o"

    assert settings.model_id == "openai/gpt-4o"


def test_model_id_raises_when_provider_is_unset():
    """``model_id`` raises when the provider is not configured."""
    settings = Settings.default()
    settings.model_name = "gpt-4o"

    with pytest.raises(ValueError):
        _ = settings.model_id


def test_model_id_raises_when_model_name_is_unset():
    """``model_id`` raises when the model name is not configured."""
    settings = Settings.default()
    settings.provider = "openai"

    with pytest.raises(ValueError):
        _ = settings.model_id


def test_provider_accepts_a_valid_litellm_provider():
    """Setting a recognized LiteLLM provider name stores it lowercased."""
    settings = Settings.default()

    settings.provider = "openai"

    assert settings.provider == "openai"


def test_provider_is_case_insensitive():
    """A provider name is matched against LiteLLM's list case-insensitively."""
    settings = Settings.default()

    settings.provider = "OpenAI"

    assert settings.provider == "openai"


def test_provider_accepts_none_to_unset():
    """Setting the provider to ``None`` unsets it without validation."""
    settings = Settings.default()
    settings.provider = "openai"

    settings.provider = None

    assert settings.provider is None


def test_provider_rejects_unknown_provider():
    """An unrecognized provider name raises, leaving the previous value intact."""
    settings = Settings.default()
    settings.provider = "openai"

    with pytest.raises(ValueError):
        settings.provider = "not-a-real-provider"

    assert settings.provider == "openai"


def test_litellm_provider_names_is_not_empty():
    """The LiteLLM-derived provider set is populated and includes known providers."""
    assert "openai" in LITELLM_PROVIDER_NAMES
    assert "anthropic" in LITELLM_PROVIDER_NAMES
    assert "ollama_chat" in LITELLM_PROVIDER_NAMES
