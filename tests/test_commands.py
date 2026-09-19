"""Tests for the slash commands and the argument coercion they share.

Commands are the only place a user changes the running configuration, and each one has
to do three things together: change the live objects, persist the change as a global
default, and refresh what the toolbar shows.
"""

from pathlib import Path

import pytest
import yaml
from fakes import RecordingApp, ScriptedClient

import arancio.storage.manager as storage_module
from arancio.commands.base import BaseCommand
from arancio.commands.cd import CdCommand
from arancio.commands.clear import ClearCommand
from arancio.commands.effort import EffortCommand
from arancio.commands.exit import ExitCommand
from arancio.commands.hello_world import HelloWorldCommand
from arancio.commands.model import ModelCommand
from arancio.commands.permissions import PermissionsCommand
from arancio.commands.provider import ProviderCommand
from arancio.commands.registry import COMMAND_REGISTRY
from arancio.commands.resume import ResumeCommand
from arancio.core.agents import Agent
from arancio.core.messages import UserMessage
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.sessions.manager import SessionManager
from arancio.sessions.registry import SessionRegistryEntry
from arancio.sessions.session import SessionConfiguration
from arancio.settings.manager import SettingsManager
from arancio.settings.settings import Settings


@pytest.fixture
def configured(settings_manager: SettingsManager) -> SettingsManager:
    """Return a settings manager loaded with a usable provider and model.

    Args:
        settings_manager: the manager under test.

    Returns:
        The same manager, loaded from a complete settings file.
    """
    data = Settings.default().to_dict()
    data.update(provider="openai", model_name="gpt-4o")
    storage_module.ARANCIO_SETTINGS_FILE.write_text(yaml.safe_dump(data))
    settings_manager.load()
    return settings_manager


@pytest.fixture
def app(tmp_path: Path) -> RecordingApp:
    """Return a fake app rooted in the test's temporary directory.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A recording app.
    """
    return RecordingApp(working_directory=tmp_path)


def _persisted() -> dict:
    """Read back what a command persisted to the settings file.

    Returns:
        The parsed settings mapping.
    """
    return yaml.safe_load(storage_module.ARANCIO_SETTINGS_FILE.read_text())


def test_every_registered_name_resolves_to_a_command() -> None:
    """The registry is the only lookup, so a broken entry breaks the command."""
    assert COMMAND_REGISTRY["quit"] is ExitCommand
    assert all(
        issubclass(command, BaseCommand) for command in COMMAND_REGISTRY.values()
    )
    assert {command.name for command in COMMAND_REGISTRY.values()} <= set(
        COMMAND_REGISTRY
    )


def test_a_typed_argument_is_coerced_from_its_prompt_word() -> None:
    """Prompt words are strings, so an ``int`` parameter needs converting."""
    assert HelloWorldCommand.run(name="Ada", times="2") == (
        "Hello World, Ada\nHello World, Ada"
    )


def test_an_uncoercible_argument_is_reported_by_name() -> None:
    """The message names the argument, so the user knows which word to fix."""
    with pytest.raises(TypeError, match="times"):
        HelloWorldCommand.run(name="Ada", times="lots")


def test_cd_moves_the_working_directory(app: RecordingApp, tmp_path: Path) -> None:
    """The app owns the directory; the command only asks it to move."""
    (tmp_path / "sub").mkdir()

    result = CdCommand.execute(path="sub", application=app)

    assert app.working_directory == (tmp_path / "sub").resolve()
    assert result == f"Working directory set to {app.working_directory}"


def test_cd_into_a_missing_directory_is_refused(app: RecordingApp) -> None:
    """The executor turns this into an error message rather than crashing."""
    with pytest.raises(ValueError):
        CdCommand.execute(path="nowhere", application=app)


def test_model_applies_and_persists_the_new_name(
    configured: SettingsManager, app: RecordingApp, client: ScriptedClient
) -> None:
    """One command has to change the client, the file and the toolbar together."""
    result = ModelCommand.execute(
        model_name="gpt-5", application=app, settings_manager=configured
    )

    assert result == "Model set to openai/gpt-5"
    assert client.model_id == "openai/gpt-5"
    assert _persisted()["model_name"] == "gpt-5"
    assert app.displayed_model_ids == ["openai/gpt-5"]


def test_model_without_a_provider_leaves_the_name_untouched(
    settings_manager: SettingsManager, app: RecordingApp
) -> None:
    """A half-applied change would leave the settings inconsistent."""
    with pytest.raises(ValueError):
        ModelCommand.execute(
            model_name="gpt-5", application=app, settings_manager=settings_manager
        )

    assert settings_manager.settings.model_name is None


def test_provider_is_normalized_before_it_is_stored(
    configured: SettingsManager, app: RecordingApp
) -> None:
    """The provider becomes half of a model id, so its case must be settled."""
    result = ProviderCommand.execute(
        provider="Anthropic", application=app, settings_manager=configured
    )

    assert result == "Provider set to anthropic"
    assert _persisted()["provider"] == "anthropic"


def test_an_unknown_provider_is_refused(
    configured: SettingsManager, app: RecordingApp
) -> None:
    """LiteLLM resolves credentials from the provider, so a typo must not stick."""
    with pytest.raises(ValueError):
        ProviderCommand.execute(
            provider="not-a-provider", application=app, settings_manager=configured
        )


def test_effort_accepts_the_word_null_as_no_thinking(
    configured: SettingsManager, app: RecordingApp
) -> None:
    """There is no other way to type "none" into a text argument."""
    result = EffortCommand.execute(
        level="NULL", application=app, settings_manager=configured
    )

    assert result == "Thinking effort set to null"
    assert configured.settings.thinking_effort is None
    assert app.displayed_efforts == [None]


def test_effort_without_a_model_changes_nothing(
    settings_manager: SettingsManager, app: RecordingApp
) -> None:
    """Effort belongs to a model, so it is refused before anything is written."""
    with pytest.raises(ValueError):
        EffortCommand.execute(
            level="high", application=app, settings_manager=settings_manager
        )


def test_permissions_without_a_level_reports_the_current_one(
    configured: SettingsManager,
) -> None:
    """The read form is how a user checks what the agent may do."""
    result = PermissionsCommand.execute(category="READ", settings_manager=configured)

    assert result == "read permission level: ask"


def test_permissions_reports_a_revoked_category_differently(
    configured: SettingsManager,
) -> None:
    """Saying the level is none would mislead: the tools do not exist at all."""
    configured.settings.permissions[PermissionCategory.WEB] = PermissionLevel.NONE

    assert (
        PermissionsCommand.execute(category="web", settings_manager=configured)
        == "No web permission set"
    )


def test_permissions_sets_and_persists_a_level(configured: SettingsManager) -> None:
    """A granted level has to survive into the next session."""
    result = PermissionsCommand.execute(
        category="write", level="AUTO", settings_manager=configured
    )

    assert result == "write permission level set to auto"
    assert configured.settings.permissions[PermissionCategory.WRITE] is (
        PermissionLevel.AUTO
    )
    assert _persisted()["permissions"]["write"] == "auto"


def test_permissions_removes_a_grant_with_the_word_null(
    configured: SettingsManager,
) -> None:
    """Revoking is what stops the tools being built at all."""
    result = PermissionsCommand.execute(
        category="execute", level="null", settings_manager=configured
    )

    assert result == "execute permission removed"
    assert _persisted()["permissions"]["execute"] is None


@pytest.mark.parametrize(
    ("category", "level"),
    [("netwrk", "auto"), ("read", "sometimes")],
    ids=["bad-category", "bad-level"],
)
def test_permissions_rejects_what_it_cannot_resolve(
    configured: SettingsManager, category: str, level: str
) -> None:
    """The error lists the valid options, since the user typed a free word."""
    with pytest.raises(ValueError, match="Valid"):
        PermissionsCommand.execute(
            category=category, level=level, settings_manager=configured
        )


def test_clear_starts_a_new_chat_everywhere_at_once(
    configured: SettingsManager,
    app: RecordingApp,
    agent: Agent,
    session_manager: SessionManager,
    tmp_path: Path,
) -> None:
    """Four things hold chat state, and ``/clear`` has to reset all four."""
    first = session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )
    agent.add_message_to_history(UserMessage(content="earlier"))
    configured.settings.model_name = "gpt-5"

    ClearCommand.execute(
        agent=agent,
        application=app,
        settings_manager=configured,
        session_manager=session_manager,
    )

    assert session_manager.require_current() is not first
    assert configured.settings.model_name == "gpt-4o"
    assert app.cleared == 1


def test_resume_by_exact_id_restores_history_configuration_and_ui(
    configured: SettingsManager,
    app: RecordingApp,
    agent: Agent,
    session_manager: SessionManager,
    tmp_path: Path,
) -> None:
    """A resumed session has to look, from every angle, like it never left."""
    first = session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )
    session_manager.session_recorder.message(UserMessage(content="from the first chat"))

    configured.settings.model_name = "gpt-5"
    session_manager.discard_and_create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )
    agent.add_message_to_history(UserMessage(content="from the second chat"))

    result = ResumeCommand.execute(
        query=first.id,
        application=app,
        agent=agent,
        settings_manager=configured,
        session_manager=session_manager,
    )

    assert result is None
    assert session_manager.require_current().id == first.id
    assert configured.settings.model_name == "gpt-4o"
    assert agent._message_history == [UserMessage(content="from the first chat")]
    assert app.cleared == 1
    assert app.populated == 1
    assert app.displayed_model_ids == ["openai/gpt-4o"]


def test_resume_by_a_name_fragment_finds_the_same_session(
    configured: SettingsManager,
    app: RecordingApp,
    agent: Agent,
    session_manager: SessionManager,
    tmp_path: Path,
) -> None:
    """A user rarely remembers a session's full ID."""
    first = session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )

    result = ResumeCommand.execute(
        query=first.id[:8],
        application=app,
        agent=agent,
        settings_manager=configured,
        session_manager=session_manager,
    )

    assert result is None
    assert session_manager.require_current().id == first.id


def test_resume_with_no_match_is_refused(
    configured: SettingsManager,
    app: RecordingApp,
    agent: Agent,
    session_manager: SessionManager,
) -> None:
    """A typo must not be silently ignored."""
    with pytest.raises(ValueError, match="No session matches"):
        ResumeCommand.execute(
            query="nosuchsession",
            application=app,
            agent=agent,
            settings_manager=configured,
            session_manager=session_manager,
        )


def test_resume_with_more_than_one_match_lists_them_instead_of_resuming(
    configured: SettingsManager,
    app: RecordingApp,
    agent: Agent,
    session_manager: SessionManager,
    tmp_path: Path,
) -> None:
    """Guessing which one the user meant would be worse than asking."""
    first = session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )
    session_manager.registry.entries.append(
        SessionRegistryEntry(
            id="second",
            name=f"{first.name}-copy",
            working_directory=tmp_path,
            configuration=SessionConfiguration.from_settings(configured.settings),
            log_path=tmp_path / "second.jsonl",
            status="healthy",
        )
    )

    result = ResumeCommand.execute(
        query=first.id[:8],
        application=app,
        agent=agent,
        settings_manager=configured,
        session_manager=session_manager,
    )

    assert result is not None
    assert first.id in result
    assert "second" in result
    assert session_manager.require_current().id == first.id
    assert app.cleared == 0


def test_resume_from_a_different_directory_moves_to_it(
    configured: SettingsManager,
    agent: Agent,
    session_manager: SessionManager,
    tmp_path: Path,
) -> None:
    """Resuming a session from another project follows it there."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    elsewhere_session = session_manager.create(
        working_directory=elsewhere,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )
    session_manager.discard_and_create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )
    app = RecordingApp(working_directory=tmp_path)

    result = ResumeCommand.execute(
        query=elsewhere_session.id,
        application=app,
        agent=agent,
        settings_manager=configured,
        session_manager=session_manager,
    )

    assert result is None
    assert session_manager.require_current().id == elsewhere_session.id
    assert app.working_directory == elsewhere.resolve()


def test_resume_moves_to_the_resumed_directory_even_when_unchanged(
    configured: SettingsManager,
    app: RecordingApp,
    agent: Agent,
    session_manager: SessionManager,
    tmp_path: Path,
) -> None:
    """The move always happens, so nothing else has to special-case a no-op."""
    first = session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )
    session_manager.discard_and_create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )

    result = ResumeCommand.execute(
        query=first.id,
        application=app,
        agent=agent,
        settings_manager=configured,
        session_manager=session_manager,
    )

    assert result is None
    assert session_manager.require_current().id == first.id
    assert app.working_directory == tmp_path.resolve()


def test_exit_asks_the_app_to_quit(app: RecordingApp) -> None:
    """Quitting is the app's to do; the command only requests it."""
    assert ExitCommand.execute(application=app) is None
    assert app.exited is True
