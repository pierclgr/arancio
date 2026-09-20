"""Tests for the slash commands and the argument coercion they share.

Commands are the only place a user changes the running configuration, and each one has
to do three things together: change the live objects, persist the change as a global
default, and refresh what the toolbar shows.
"""

import json
import re
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
from arancio.commands.fork import ForkCommand
from arancio.commands.hello_world import HelloWorldCommand
from arancio.commands.model import ModelCommand
from arancio.commands.permissions import PermissionsCommand
from arancio.commands.provider import ProviderCommand
from arancio.commands.registry import (
    COMMAND_REGISTRY,
    SESSION_DISCARDING_COMMANDS,
)
from arancio.commands.rename import RenameCommand
from arancio.commands.resume import ResumeCommand
from arancio.core.agents import Agent
from arancio.core.messages import UserMessage
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.sessions.manager import SessionManager
from arancio.sessions.registry import SessionRegistryEntry
from arancio.sessions.session import SessionConfiguration
from arancio.settings.manager import SettingsManager
from arancio.settings.settings import Settings
from arancio.storage.manager import StorageManager


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


def _records(session_manager: SessionManager) -> list[dict]:
    """Read every record written to the open session's log.

    Args:
        session_manager: the manager owning the open chat.

    Returns:
        One decoded record per line.
    """
    path = session_manager.current.path
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_every_registered_name_resolves_to_a_command() -> None:
    """The registry is the only lookup, so a broken entry breaks the command."""
    assert COMMAND_REGISTRY["quit"] is ExitCommand
    assert COMMAND_REGISTRY["new"] is ClearCommand
    assert all(
        issubclass(command, BaseCommand) for command in COMMAND_REGISTRY.values()
    )
    assert {command.name for command in COMMAND_REGISTRY.values()} <= set(
        COMMAND_REGISTRY
    )


def test_every_discarding_command_is_a_registered_command() -> None:
    """The executor keys on the class, so an unregistered one could never run."""
    assert SESSION_DISCARDING_COMMANDS <= set(COMMAND_REGISTRY.values())


def test_a_typed_argument_is_coerced_from_its_prompt_word() -> None:
    """Prompt words are strings, so an ``int`` parameter needs converting."""
    assert HelloWorldCommand.run(name="Ada", times="2") == (
        "Hello World, Ada\nHello World, Ada"
    )


def test_an_uncoercible_argument_is_reported_by_name() -> None:
    """The message names the argument, so the user knows which word to fix."""
    with pytest.raises(TypeError, match="times"):
        HelloWorldCommand.run(name="Ada", times="lots")


def test_cd_moves_the_working_directory(
    app: RecordingApp,
    session_manager: SessionManager,
    configured: SettingsManager,
    tmp_path: Path,
) -> None:
    """The app owns the directory; the command mirrors it onto the session."""
    (tmp_path / "sub").mkdir()
    session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )

    result = CdCommand.execute(
        path="sub", application=app, session_manager=session_manager
    )

    assert app.working_directory == (tmp_path / "sub").resolve()
    assert result == f"Working directory set to {app.working_directory}"
    assert session_manager.current.working_directory == app.working_directory
    assert _records(session_manager)[-1]["type"] == "state_changed"


def test_cd_into_a_missing_directory_is_refused(
    app: RecordingApp, session_manager: SessionManager
) -> None:
    """The executor turns this into an error message rather than crashing."""
    with pytest.raises(ValueError):
        CdCommand.execute(
            path="nowhere", application=app, session_manager=session_manager
        )


def test_model_applies_and_persists_the_new_name(
    configured: SettingsManager,
    app: RecordingApp,
    client: ScriptedClient,
    session_manager: SessionManager,
    tmp_path: Path,
) -> None:
    """One command has to change the client, the file and the toolbar together."""
    session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )

    result = ModelCommand.execute(
        model_name="gpt-5",
        application=app,
        settings_manager=configured,
        session_manager=session_manager,
    )

    assert result == "Model set to openai/gpt-5"
    assert client.model_id == "openai/gpt-5"
    assert _persisted()["model_name"] == "gpt-5"
    assert app.displayed_model_ids == ["openai/gpt-5"]
    assert session_manager.current.configuration.model_name == "gpt-5"
    assert _records(session_manager)[-1]["type"] == "state_changed"


def test_model_without_a_provider_leaves_the_name_untouched(
    settings_manager: SettingsManager,
    app: RecordingApp,
    session_manager: SessionManager,
) -> None:
    """A half-applied change would leave the settings inconsistent."""
    with pytest.raises(ValueError):
        ModelCommand.execute(
            model_name="gpt-5",
            application=app,
            settings_manager=settings_manager,
            session_manager=session_manager,
        )

    assert settings_manager.settings.model_name is None


def test_provider_is_normalized_before_it_is_stored(
    configured: SettingsManager,
    app: RecordingApp,
    session_manager: SessionManager,
    tmp_path: Path,
) -> None:
    """The provider becomes half of a model id, so its case must be settled."""
    session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )

    result = ProviderCommand.execute(
        provider="Anthropic",
        application=app,
        settings_manager=configured,
        session_manager=session_manager,
    )

    assert result == "Provider set to anthropic"
    assert _persisted()["provider"] == "anthropic"
    assert session_manager.current.configuration.provider == "anthropic"
    assert _records(session_manager)[-1]["type"] == "state_changed"


def test_an_unknown_provider_is_refused(
    configured: SettingsManager, app: RecordingApp, session_manager: SessionManager
) -> None:
    """LiteLLM resolves credentials from the provider, so a typo must not stick."""
    with pytest.raises(ValueError):
        ProviderCommand.execute(
            provider="not-a-provider",
            application=app,
            settings_manager=configured,
            session_manager=session_manager,
        )


def test_effort_accepts_the_word_null_as_no_thinking(
    configured: SettingsManager,
    app: RecordingApp,
    session_manager: SessionManager,
    tmp_path: Path,
) -> None:
    """There is no other way to type "none" into a text argument."""
    session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )

    result = EffortCommand.execute(
        level="NULL",
        application=app,
        settings_manager=configured,
        session_manager=session_manager,
    )

    assert result == "Thinking effort set to null"
    assert configured.settings.thinking_effort is None
    assert app.displayed_efforts == [None]
    assert session_manager.current.configuration.thinking_effort is None
    assert _records(session_manager)[-1]["type"] == "state_changed"


def test_effort_without_a_model_changes_nothing(
    settings_manager: SettingsManager,
    app: RecordingApp,
    session_manager: SessionManager,
) -> None:
    """Effort belongs to a model, so it is refused before anything is written."""
    with pytest.raises(ValueError):
        EffortCommand.execute(
            level="high",
            application=app,
            settings_manager=settings_manager,
            session_manager=session_manager,
        )


def test_permissions_without_a_level_reports_the_current_one(
    configured: SettingsManager, session_manager: SessionManager
) -> None:
    """The read form is how a user checks what the agent may do."""
    result = PermissionsCommand.execute(
        category="READ", settings_manager=configured, session_manager=session_manager
    )

    assert result == "read permission level: ask"


def test_permissions_reports_a_revoked_category_differently(
    configured: SettingsManager, session_manager: SessionManager
) -> None:
    """Saying the level is none would mislead: the tools do not exist at all."""
    configured.settings.permissions[PermissionCategory.WEB] = PermissionLevel.NONE

    assert (
        PermissionsCommand.execute(
            category="web",
            settings_manager=configured,
            session_manager=session_manager,
        )
        == "No web permission set"
    )


def test_permissions_sets_and_persists_a_level(
    configured: SettingsManager, session_manager: SessionManager, tmp_path: Path
) -> None:
    """A granted level has to survive into the next session."""
    session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )

    result = PermissionsCommand.execute(
        category="write",
        level="AUTO",
        settings_manager=configured,
        session_manager=session_manager,
    )

    assert result == "write permission level set to auto"
    assert configured.settings.permissions[PermissionCategory.WRITE] is (
        PermissionLevel.AUTO
    )
    assert _persisted()["permissions"]["write"] == "auto"
    assert (
        session_manager.current.configuration.permissions[PermissionCategory.WRITE]
        is PermissionLevel.AUTO
    )
    assert _records(session_manager)[-1]["type"] == "state_changed"


def test_permissions_removes_a_grant_with_the_word_null(
    configured: SettingsManager, session_manager: SessionManager, tmp_path: Path
) -> None:
    """Revoking is what stops the tools being built at all."""
    session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )

    result = PermissionsCommand.execute(
        category="execute",
        level="null",
        settings_manager=configured,
        session_manager=session_manager,
    )

    assert result == "execute permission removed"
    assert _persisted()["permissions"]["execute"] is None
    assert (
        session_manager.current.configuration.permissions[PermissionCategory.EXECUTE]
        is PermissionLevel.NONE
    )
    assert _records(session_manager)[-1]["type"] == "state_changed"


@pytest.mark.parametrize(
    ("category", "level"),
    [("netwrk", "auto"), ("read", "sometimes")],
    ids=["bad-category", "bad-level"],
)
def test_permissions_rejects_what_it_cannot_resolve(
    configured: SettingsManager,
    session_manager: SessionManager,
    category: str,
    level: str,
) -> None:
    """The error lists the valid options, since the user typed a free word."""
    with pytest.raises(ValueError, match="Valid"):
        PermissionsCommand.execute(
            category=category,
            level=level,
            settings_manager=configured,
            session_manager=session_manager,
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

    assert session_manager.current is not first
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
    assert session_manager.current.id == first.id
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
    # only a session whose log is on disk is in the registry to be found
    session_manager.session_recorder.flush()
    session_manager.discard_and_create(
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
    assert session_manager.current.id == first.id


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
    # only a session whose log is on disk is in the registry to be found
    session_manager.session_recorder.flush()
    session_manager.registry.entries.append(
        SessionRegistryEntry(
            id="second",
            explicit_name=f"{first.name}-copy",
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
    assert session_manager.current.id == first.id
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
    # only a session whose log is on disk is in the registry to be found
    session_manager.session_recorder.flush()
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
    assert session_manager.current.id == elsewhere_session.id
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
    # only a session whose log is on disk is in the registry to be found
    session_manager.session_recorder.flush()
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
    assert session_manager.current.id == first.id
    assert app.working_directory == tmp_path.resolve()


def test_rename_updates_the_session_and_its_registry_entry(
    configured: SettingsManager, session_manager: SessionManager, tmp_path: Path
) -> None:
    """The active session's own registry entry must reflect a rename immediately."""
    session = session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )

    result = RenameCommand.execute(new_name="demo", session_manager=session_manager)

    assert result == "Session renamed to 'demo'"
    assert session.name == "demo"
    assert session_manager.registry.get(session.id).name == "demo"


def test_a_rename_survives_a_reload(
    configured: SettingsManager,
    session_manager: SessionManager,
    storage_manager: StorageManager,
    tmp_path: Path,
) -> None:
    """The renamed session must come back under its new name, not its old one."""
    session = session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )
    RenameCommand.execute(new_name="demo", session_manager=session_manager)

    reopened = SessionManager(
        storage_manager,
        SessionConfiguration.from_settings(Settings.default()),
        root=storage_manager.root,
    )
    restored = reopened.load(session.id)

    assert restored.name == "demo"
    assert reopened.registry.get(session.id).name == "demo"


def test_fork_renames_a_named_source_to_main_and_names_the_child(
    configured: SettingsManager, session_manager: SessionManager, tmp_path: Path
) -> None:
    """A named, non-fork source becomes ``:main``; the child gets a fork suffix."""
    first = session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )
    RenameCommand.execute(new_name="demo", session_manager=session_manager)

    result = ForkCommand.execute(session_manager=session_manager)

    forked = session_manager.current
    assert forked is not first
    assert first.name == "demo:main"
    assert re.fullmatch(r"demo:fork_\d{14}", forked.name)
    assert result == f"Session {first.name} forked to {forked.name}"
    assert forked.forked_from == first.id


def test_forking_a_fork_does_not_rename_it_again_and_reuses_the_base_name(
    configured: SettingsManager, session_manager: SessionManager, tmp_path: Path
) -> None:
    """Re-forking a fork must not rename it or pile up ``:fork_`` suffixes."""
    session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )
    RenameCommand.execute(new_name="demo", session_manager=session_manager)
    ForkCommand.execute(session_manager=session_manager)
    first_fork = session_manager.current
    first_fork_name = first_fork.name

    ForkCommand.execute(session_manager=session_manager)

    second_fork = session_manager.current
    assert first_fork.name == first_fork_name
    assert re.fullmatch(r"demo:fork_\d{14}", second_fork.name)


def test_forking_an_unnamed_session_leaves_names_untouched(
    configured: SettingsManager, session_manager: SessionManager, tmp_path: Path
) -> None:
    """Without a name, a fork keeps behaving exactly as before this convention."""
    first = session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )

    result = ForkCommand.execute(session_manager=session_manager)

    forked = session_manager.current
    assert forked is not first
    assert forked.name == forked.id
    assert result == f"Session {first.id} forked to {forked.id}"


def test_fork_copies_the_source_session_s_events_under_a_new_id(
    configured: SettingsManager, session_manager: SessionManager, tmp_path: Path
) -> None:
    """Forking must duplicate the source's events onto a fresh identity."""
    source = session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )
    session_manager.session_recorder.message(UserMessage(content="hi"))
    events_before_fork = [event.record for event in source.events[1:]]

    ForkCommand.execute(session_manager=session_manager)

    forked = session_manager.current
    assert forked.id != source.id
    assert [event.record for event in forked.events[1:]] == events_before_fork
    header = json.loads(forked.path.read_text().splitlines()[0])
    assert header["forked_from"] == source.id


def test_fork_confirmation_is_also_recorded_into_the_source(
    configured: SettingsManager, session_manager: SessionManager, tmp_path: Path
) -> None:
    """The source's replayed log must show the fork happened, not just the fork's.

    ``ForkCommand`` only owns the write into the source: the write into the fork itself
    (now the open chat) is the executor's generic post-command write, exercised
    separately in ``tests/test_executor.py``.
    """
    source = session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )
    session_manager.session_recorder.message(UserMessage(content="hi"))

    result = ForkCommand.execute(session_manager=session_manager)

    assert result in [event.record.get("content") for event in source.events]
    saved_lines = source.path.read_text().splitlines()
    assert any(json.loads(line).get("content") == result for line in saved_lines)


def test_forking_a_chat_that_was_never_saved_writes_no_log(
    configured: SettingsManager, session_manager: SessionManager, tmp_path: Path
) -> None:
    """A fork of nothing is still a switch, but it has nothing to persist."""
    source = session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )

    ForkCommand.execute(session_manager=session_manager)

    forked = session_manager.current
    assert forked is not source
    assert forked.forked_from == source.id
    assert not source.path.exists()
    assert not forked.path.exists()


def test_fork_copies_file_read_state_without_aliasing_it(
    configured: SettingsManager, session_manager: SessionManager, tmp_path: Path
) -> None:
    """The read-first guard state must carry over without being aliased."""
    source = session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )
    session_manager.session_recorder.file_state("/tmp/a", 1.0)

    ForkCommand.execute(session_manager=session_manager)

    forked = session_manager.current
    assert forked.file_states == source.file_states
    forked.file_states["/tmp/b"] = 2.0
    assert "/tmp/b" not in source.file_states


def test_a_fork_survives_a_reload_with_its_provenance_intact(
    configured: SettingsManager,
    session_manager: SessionManager,
    storage_manager: StorageManager,
    tmp_path: Path,
) -> None:
    """A forked session's provenance and history must be durable, not just live."""
    session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )
    session_manager.session_recorder.message(UserMessage(content="hi"))
    source_history = session_manager.model_history()
    source_id = session_manager.current.id

    ForkCommand.execute(session_manager=session_manager)
    forked = session_manager.current

    reopened = SessionManager(
        storage_manager,
        SessionConfiguration.from_settings(Settings.default()),
        root=storage_manager.root,
    )
    restored = reopened.load(forked.id)

    assert restored.forked_from == source_id
    assert reopened.model_history() == source_history


def test_fork_leaves_the_live_agent_and_ui_untouched(
    configured: SettingsManager,
    app: RecordingApp,
    agent: Agent,
    session_manager: SessionManager,
    tmp_path: Path,
) -> None:
    """Forking duplicates persisted identity only; nothing visible resets."""
    session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )
    agent.add_message_to_history(UserMessage(content="earlier"))

    ForkCommand.execute(session_manager=session_manager)

    assert agent._message_history == [UserMessage(content="earlier")]
    assert app.cleared == 0
    assert app.populated == 0


def test_resume_by_the_shared_base_name_finds_a_source_and_its_fork(
    configured: SettingsManager, session_manager: SessionManager, tmp_path: Path
) -> None:
    """A named source and its fork share a base name a query can find both by."""
    source = session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(configured.settings),
    )
    RenameCommand.execute(new_name="demo", session_manager=session_manager)

    ForkCommand.execute(session_manager=session_manager)
    forked = session_manager.current

    matches = {entry.id for entry in session_manager.registry.find("demo")}
    assert matches == {source.id, forked.id}


def test_exit_asks_the_app_to_quit(app: RecordingApp) -> None:
    """Quitting is the app's to do; the command only requests it."""
    assert ExitCommand.execute(application=app) is None
    assert app.exited is True
