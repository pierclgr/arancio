"""Tests for the settings object, its validator and the manager that applies it.

Two things matter here. A broken settings file must never stop the app: every bad field
falls back and says so. And the manager keeps two snapshots — the global defaults on
disk and the active session's overrides — which is what lets ``/clear`` and session
restore put things back.
"""

from typing import Any

import pytest
import yaml
from fakes import ScriptedClient

import arancio.storage.manager as storage_module
from arancio.core.agents import Agent
from arancio.core.messages import ErrorMessage, Message, WarningMessage
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.sessions.session import SessionConfiguration
from arancio.settings.manager import SettingsManager
from arancio.settings.settings import Settings
from arancio.settings.validator import SettingsValidator


def _complete(**overrides: Any) -> dict:
    """Build a valid settings mapping with the given fields replaced.

    Args:
        **overrides: fields to override on the default mapping.

    Returns:
        A settings dictionary ready for the validator.
    """
    data = Settings.default().to_dict()
    data.update(provider="openai", model_name="gpt-4o")
    data.update(overrides)
    return data


def _texts(messages: list[Message]) -> list[str]:
    """Return the content of each message.

    Args:
        messages: the messages to read.

    Returns:
        One content string per message.
    """
    return [message.content for message in messages]


def test_a_provider_is_stored_lowercased() -> None:
    """The value is used to build a model id, so its case cannot vary."""
    settings = Settings.default()

    settings.provider = "OpenAI"

    assert settings.provider == "openai"


def test_an_unknown_provider_is_refused_on_assignment() -> None:
    """Catching it here is what lets ``/provider`` report a clear error."""
    settings = Settings.default()

    with pytest.raises(ValueError):
        settings.provider = "not-a-provider"


def test_the_two_halves_of_a_model_id_are_reported_separately() -> None:
    """``/model`` and the executor each need to know which half is missing."""
    settings = Settings.default()

    with pytest.raises(ValueError, match="provider"):
        settings.model_id

    settings.provider = "openai"
    with pytest.raises(ValueError, match="model name"):
        settings.model_id

    settings.model_name = "gpt-4o"
    assert settings.model_id == "openai/gpt-4o"


def test_settings_round_trip_through_their_dictionary_form() -> None:
    """The dictionary form is what reaches disk, so it must be lossless."""
    settings = Settings.default()
    settings.provider = "openai"
    settings.model_name = "gpt-4o"

    assert Settings.from_dict(settings.to_dict()) == settings


def test_a_missing_field_quietly_takes_its_default() -> None:
    """An older settings file must keep working after a new field is added."""
    data = _complete()
    del data["max_retries"]

    settings, messages = SettingsValidator.validate(data)

    assert settings.max_retries == 5
    assert messages == []


@pytest.mark.parametrize(
    ("field", "bad_value", "expected"),
    [
        ("max_turns", 0, "inf"),
        ("max_turns", "many", "inf"),
        ("max_retries", -1, 5),
        ("turn_wait_time", "soon", 3.0),
        ("turn_wait_time_multiplier", -2, 2.0),
        ("thinking_effort", 7, "medium"),
    ],
)
def test_an_invalid_field_falls_back_with_a_warning(
    field: str, bad_value: Any, expected: Any
) -> None:
    """A bad value is a warning because the field has a usable default."""
    settings, messages = SettingsValidator.validate(_complete(**{field: bad_value}))

    assert getattr(settings, field) == expected
    assert isinstance(messages[0], WarningMessage)
    assert messages[0].content == (
        f'settings.yml: invalid value {bad_value!r} for "{field}"; '
        f"using default {expected!r}."
    )


@pytest.mark.parametrize("field", ["provider", "model_name"])
def test_an_unconfigured_model_field_is_an_error_with_its_own_wording(
    field: str,
) -> None:
    """These two have no usable default, so the app cannot talk to a model.

    "is not set" reads differently from "invalid value", and it has to: a first run is
    not the same problem as a typo.
    """
    settings, messages = SettingsValidator.validate(_complete(**{field: None}))

    assert getattr(settings, field) is None
    assert isinstance(messages[0], ErrorMessage)
    assert messages[0].content == f'settings.yml: "{field}" is not set.'


def test_an_unknown_top_level_key_is_dropped_with_a_warning() -> None:
    """A typo in a field name would otherwise be silently ignored forever."""
    _, messages = SettingsValidator.validate(_complete(mdel_name="gpt-4o"))

    assert "settings.yml: unknown field 'mdel_name'; entry ignored." in _texts(messages)


def test_permissions_always_come_back_complete() -> None:
    """The permission manager needs all four categories, so a partial file is filled."""
    settings, messages = SettingsValidator.validate(
        _complete(permissions={"read": "auto"})
    )

    assert settings.permissions[PermissionCategory.READ] is PermissionLevel.AUTO
    assert settings.permissions[PermissionCategory.WRITE] is PermissionLevel.ASK
    assert messages == []


def test_an_explicit_null_permission_level_is_a_real_value() -> None:
    """``null`` means the category is revoked, which is a choice, not a mistake."""
    settings, messages = SettingsValidator.validate(
        _complete(permissions={"web": None})
    )

    assert settings.permissions[PermissionCategory.WEB] is PermissionLevel.NONE
    assert messages == []


def test_an_unknown_permission_category_is_dropped() -> None:
    """There is no category to default it to, so it is reported and ignored."""
    _, messages = SettingsValidator.validate(_complete(permissions={"netwrk": "auto"}))

    assert (
        "settings.yml: unknown permission category 'netwrk'; entry ignored."
        in _texts(messages)
    )


def test_an_invalid_permission_level_falls_back_to_asking() -> None:
    """Asking is the safe default when the file says something unreadable."""
    settings, messages = SettingsValidator.validate(
        _complete(permissions={"execute": "sometimes"})
    )

    assert settings.permissions[PermissionCategory.EXECUTE] is PermissionLevel.ASK
    assert isinstance(messages[0], WarningMessage)


def test_loading_pushes_the_file_into_the_live_objects(
    settings_manager: SettingsManager, client: ScriptedClient, agent: Agent
) -> None:
    """Settings are only real once the client and agent are actually using them."""
    storage_module.ARANCIO_SETTINGS_FILE.write_text(
        yaml.safe_dump(_complete(max_retries=9, turn_wait_time=0.5))
    )

    settings, messages = settings_manager.load()

    assert messages == []
    assert settings.model_id == "openai/gpt-4o"
    assert client.model_id == "openai/gpt-4o"
    assert agent.max_retries == 9
    assert agent.retry_delay == 0.5


def test_an_unconfigured_model_leaves_the_client_without_one(
    settings_manager: SettingsManager, client: ScriptedClient
) -> None:
    """A first run must still start, just without being able to send anything."""
    storage_module.ARANCIO_SETTINGS_FILE.write_text(
        yaml.safe_dump(Settings.default().to_dict())
    )

    _, messages = settings_manager.load()

    assert client.model_id is None
    assert len(messages) == 2


def test_a_session_override_does_not_disturb_the_global_default(
    settings_manager: SettingsManager,
) -> None:
    """A chat can change its model without rewriting what every new chat starts from."""
    storage_module.ARANCIO_SETTINGS_FILE.write_text(yaml.safe_dump(_complete()))
    settings_manager.load()

    settings_manager.apply_session_configuration(
        SessionConfiguration(
            provider="anthropic",
            model_name="claude",
            thinking_effort=None,
            permissions={c: PermissionLevel.AUTO for c in PermissionCategory},
        )
    )

    assert settings_manager.settings.model_id == "anthropic/claude"
    assert settings_manager.global_settings.model_id == "openai/gpt-4o"


def test_resetting_puts_the_global_default_back(
    settings_manager: SettingsManager,
) -> None:
    """This is the settings half of ``/clear``."""
    storage_module.ARANCIO_SETTINGS_FILE.write_text(yaml.safe_dump(_complete()))
    settings_manager.load()
    settings_manager.settings.model_name = "gpt-5"

    settings_manager.reset_to_global()

    assert settings_manager.settings.model_name == "gpt-4o"


def test_saving_one_field_promotes_only_that_field(
    settings_manager: SettingsManager,
) -> None:
    """``/model`` persists the model, not the rest of the session's overrides."""
    storage_module.ARANCIO_SETTINGS_FILE.write_text(yaml.safe_dump(_complete()))
    settings_manager.load()
    settings_manager.settings.model_name = "gpt-5"
    settings_manager.settings.thinking_effort = "low"

    settings_manager.save_model_name()

    written = yaml.safe_load(storage_module.ARANCIO_SETTINGS_FILE.read_text())
    assert written["model_name"] == "gpt-5"
    assert written["thinking_effort"] == "medium"


def test_the_two_snapshots_are_independent_objects(
    settings_manager: SettingsManager,
) -> None:
    """A shared permissions dict would let a session edit the global default."""
    storage_module.ARANCIO_SETTINGS_FILE.write_text(yaml.safe_dump(_complete()))
    settings_manager.load()

    settings_manager.settings.permissions[PermissionCategory.READ] = (
        PermissionLevel.NONE
    )

    assert (
        settings_manager.global_settings.permissions[PermissionCategory.READ]
        is PermissionLevel.ASK
    )
