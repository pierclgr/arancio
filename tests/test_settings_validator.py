"""Tests for SettingsValidator: per-field settings validation and fallback."""

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
from arancio.core.messages import ErrorMessage, WarningMessage
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.settings.settings import Settings
from arancio.settings.validator import SettingsValidator


def _valid_data():
    """Return a fully valid, fully configured settings dictionary.

    ``provider``/``model_name`` are set to a real value (not ``null``): unlike
    every other field, ``null`` is never a valid value for them (their
    default *is* ``null``, so a present ``null`` is treated as wrong, not
    missing), so a "fully valid" baseline must configure them.

    Returns:
        A settings dictionary with no invalid or unconfigured fields.
    """
    data = Settings.default().to_dict()
    data["provider"] = "openai"
    data["model_name"] = "gpt-4o"
    return data


def test_valid_data_passes_through_with_no_messages():
    """A fully valid dictionary is loaded with no fallback messages."""
    settings, messages = SettingsValidator.validate(_valid_data())

    assert settings == Settings.from_dict(_valid_data())
    assert messages == []


def test_provider_present_as_none_reports_error():
    """A ``provider`` present as ``null`` is wrong (not missing), reports an error."""
    data = _valid_data()
    data["provider"] = None

    settings, messages = SettingsValidator.validate(data)

    assert settings.provider is None
    assert len(messages) == 1
    assert isinstance(messages[0], ErrorMessage)


def test_provider_absent_key_reports_error_like_present_none():
    """A ``provider`` key absent entirely reports the same error as ``null``."""
    data = _valid_data()
    del data["provider"]

    settings, messages = SettingsValidator.validate(data)

    assert settings.provider is None
    assert len(messages) == 1
    assert isinstance(messages[0], ErrorMessage)


def test_model_name_present_as_none_reports_error():
    """A ``model_name`` present as ``null`` is wrong (not missing), reports an error."""
    data = _valid_data()
    data["model_name"] = None

    settings, messages = SettingsValidator.validate(data)

    assert settings.model_name is None
    assert len(messages) == 1
    assert isinstance(messages[0], ErrorMessage)


def test_model_name_absent_key_reports_error_like_present_none():
    """A ``model_name`` key absent entirely reports the same error as ``null``."""
    data = _valid_data()
    del data["model_name"]

    settings, messages = SettingsValidator.validate(data)

    assert settings.model_name is None
    assert len(messages) == 1
    assert isinstance(messages[0], ErrorMessage)


def test_missing_field_falls_back_silently():
    """A field absent from the dictionary silently takes its default."""
    data = _valid_data()
    del data["max_turns"]

    settings, messages = SettingsValidator.validate(data)

    assert settings.max_turns == AGENT_DEFAULT_MAX_TURNS
    assert messages == []


def test_invalid_provider_reports_error_and_defaults_to_none():
    """An unknown provider falls back to ``None`` and reports an error."""
    data = _valid_data()
    data["provider"] = "notreal"

    settings, messages = SettingsValidator.validate(data)

    assert settings.provider is None
    assert len(messages) == 1
    assert isinstance(messages[0], ErrorMessage)


def test_non_str_provider_reports_error_and_defaults_to_none():
    """A non-string provider falls back to ``None`` and reports an error."""
    data = _valid_data()
    data["provider"] = 42

    settings, messages = SettingsValidator.validate(data)

    assert settings.provider is None
    assert isinstance(messages[0], ErrorMessage)


def test_valid_provider_is_accepted_case_insensitively():
    """A valid provider name in any case is accepted with no message."""
    data = _valid_data()
    data["provider"] = "OpenAI"

    settings, messages = SettingsValidator.validate(data)

    assert settings.provider == "openai"
    assert messages == []


def test_non_str_model_name_reports_error_and_defaults_to_none():
    """A non-string model name falls back to ``None`` and reports an error."""
    data = _valid_data()
    data["model_name"] = 42

    settings, messages = SettingsValidator.validate(data)

    assert settings.model_name is None
    assert isinstance(messages[0], ErrorMessage)


def test_non_str_thinking_effort_reports_warning_and_defaults():
    """A non-string thinking_effort falls back to its default with a warning."""
    data = _valid_data()
    data["thinking_effort"] = 42

    settings, messages = SettingsValidator.validate(data)

    assert settings.thinking_effort == LITELLM_DEFAULT_THINKING_EFFORT
    assert len(messages) == 1
    assert isinstance(messages[0], WarningMessage)


def test_none_thinking_effort_is_accepted_silently():
    """An explicit ``null`` thinking_effort (disabling thinking) is valid."""
    data = _valid_data()
    data["thinking_effort"] = None

    settings, messages = SettingsValidator.validate(data)

    assert settings.thinking_effort is None
    assert messages == []


def test_non_str_thinking_summary_reports_warning_and_defaults():
    """A non-string thinking_summary falls back to its default with a warning."""
    data = _valid_data()
    data["thinking_summary"] = 42

    settings, messages = SettingsValidator.validate(data)

    assert settings.thinking_summary == LITELLM_DEFAULT_THINKING_SUMMARY
    assert isinstance(messages[0], WarningMessage)


def test_max_turns_accepts_unlimited_marker_and_positive_int():
    """``max_turns`` accepts the unlimited marker and any positive integer."""
    data = _valid_data()
    data["max_turns"] = 7

    settings, messages = SettingsValidator.validate(data)

    assert settings.max_turns == 7
    assert messages == []


def test_max_turns_rejects_zero_negative_bool_and_non_int():
    """Invalid ``max_turns`` values fall back to the unlimited default."""
    for bad_value in (0, -1, True, "ten", 3.5):
        data = _valid_data()
        data["max_turns"] = bad_value

        settings, messages = SettingsValidator.validate(data)

        assert settings.max_turns == AGENT_DEFAULT_MAX_TURNS
        assert len(messages) == 1
        assert isinstance(messages[0], WarningMessage)


def test_max_retries_accepts_zero():
    """``max_retries`` accepts zero as a valid non-negative integer."""
    data = _valid_data()
    data["max_retries"] = 0

    settings, messages = SettingsValidator.validate(data)

    assert settings.max_retries == 0
    assert messages == []


def test_max_retries_rejects_negative_and_bool():
    """``max_retries`` falls back to its default for negative ints and bools."""
    for bad_value in (-1, True):
        data = _valid_data()
        data["max_retries"] = bad_value

        settings, messages = SettingsValidator.validate(data)

        assert settings.max_retries == AGENT_DEFAULT_MAX_RETRIES
        assert isinstance(messages[0], WarningMessage)


def test_turn_wait_time_accepts_int_and_float():
    """``turn_wait_time`` accepts both int and float non-negative values."""
    data = _valid_data()
    data["turn_wait_time"] = 5

    settings, messages = SettingsValidator.validate(data)

    assert settings.turn_wait_time == 5
    assert messages == []


def test_turn_wait_time_rejects_negative_and_bool():
    """``turn_wait_time`` falls back to its default for negative or bool values."""
    for bad_value in (-1.0, False):
        data = _valid_data()
        data["turn_wait_time"] = bad_value

        settings, messages = SettingsValidator.validate(data)

        assert settings.turn_wait_time == AGENT_DEFAULT_TURN_WAIT_TIME
        assert isinstance(messages[0], WarningMessage)


def test_turn_wait_time_multiplier_rejects_negative():
    """``turn_wait_time_multiplier`` falls back to its default when negative."""
    data = _valid_data()
    data["turn_wait_time_multiplier"] = -2.0

    settings, messages = SettingsValidator.validate(data)

    assert settings.turn_wait_time_multiplier == AGENT_DEFAULT_TURN_WAIT_TIME_MULTIPLIER
    assert isinstance(messages[0], WarningMessage)


def test_permissions_absent_key_defaults_to_all_ask_silently():
    """A missing ``permissions`` key silently defaults every category to ``ask``."""
    data = _valid_data()
    del data["permissions"]

    settings, messages = SettingsValidator.validate(data)

    assert settings.permissions == {
        category: PermissionLevel.ASK for category in PermissionCategory
    }
    assert messages == []


def test_permissions_non_dict_reports_warning_and_defaults_all():
    """A ``permissions`` value that isn't a mapping falls back to all-``ask``."""
    data = _valid_data()
    data["permissions"] = "not-a-dict"

    settings, messages = SettingsValidator.validate(data)

    assert settings.permissions == {
        category: PermissionLevel.ASK for category in PermissionCategory
    }
    assert len(messages) == 1
    assert isinstance(messages[0], WarningMessage)


def test_permissions_unknown_category_is_dropped_with_warning():
    """An unknown permission category is dropped, leaving other entries intact."""
    data = _valid_data()
    data["permissions"] = {"foo": "ask", "read": "auto"}

    settings, messages = SettingsValidator.validate(data)

    assert settings.permissions[PermissionCategory.READ] == PermissionLevel.AUTO
    # categories omitted from the (invalid) input default to ask
    assert settings.permissions[PermissionCategory.WRITE] == PermissionLevel.ASK
    assert len(messages) == 1
    assert isinstance(messages[0], WarningMessage)


def test_permissions_invalid_level_falls_back_to_ask_with_warning():
    """A known category with an invalid level falls back to ``ask``, not dropped."""
    data = _valid_data()
    data["permissions"] = {"read": "sometimes"}

    settings, messages = SettingsValidator.validate(data)

    assert settings.permissions[PermissionCategory.READ] == PermissionLevel.ASK
    assert len(messages) == 1
    assert isinstance(messages[0], WarningMessage)


def test_permissions_explicit_null_level_is_accepted_silently():
    """An explicit ``null`` level (grant removed) is a valid value, not an error."""
    data = _valid_data()
    data["permissions"] = {"read": None}

    settings, messages = SettingsValidator.validate(data)

    assert settings.permissions[PermissionCategory.READ] == PermissionLevel.NONE
    assert messages == []


def test_permissions_partial_dict_defaults_omitted_categories():
    """Categories not listed in a partial permissions dict default to ask."""
    data = _valid_data()
    data["permissions"] = {"read": "auto"}

    settings, messages = SettingsValidator.validate(data)

    assert settings.permissions == {
        PermissionCategory.READ: PermissionLevel.AUTO,
        PermissionCategory.WRITE: PermissionLevel.ASK,
        PermissionCategory.WEB: PermissionLevel.ASK,
        PermissionCategory.EXECUTE: PermissionLevel.ASK,
    }
    assert messages == []


def test_extra_unknown_top_level_key_is_dropped_with_warning():
    """An unrecognized top-level key is dropped, reported as a warning."""
    data = _valid_data()
    data["something_unexpected"] = "value"

    settings, messages = SettingsValidator.validate(data)

    assert settings == Settings.from_dict(_valid_data())
    assert len(messages) == 1
    assert isinstance(messages[0], WarningMessage)
    assert "something_unexpected" in messages[0].display_text


def test_invalid_field_message_names_field_and_default():
    """The message for an invalid field names the field and the default used."""
    data = _valid_data()
    data["max_turns"] = -5

    _, messages = SettingsValidator.validate(data)

    assert "max_turns" in messages[0].display_text
    assert repr(AGENT_DEFAULT_MAX_TURNS) in messages[0].display_text


def test_multiple_invalid_fields_report_one_message_each():
    """Several invalid fields each report their own message."""
    data = _valid_data()
    data["provider"] = "notreal"
    data["max_turns"] = -5
    data["max_retries"] = -1

    _, messages = SettingsValidator.validate(data)

    assert len(messages) == 3
