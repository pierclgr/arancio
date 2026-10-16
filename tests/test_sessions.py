"""Tests for reading chat sessions back: the codec, the log and the registry.

A session file is the only thing that survives a restart, so these tests care about two
questions. Does a message come back as the same message, and does a corrupted log stay
discoverable instead of taking the app down with it.
"""

import json
from pathlib import Path

import pytest

from arancio.core.messages import (
    AssistantChunkMessage,
    AssistantMessage,
    ErrorMessage,
    ReasoningMessage,
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
    UserMessage,
    WarningMessage,
)
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.core.tools.session import ToolSession
from arancio.sessions.checksum import SessionChecksum
from arancio.sessions.codec import message_from_record, message_to_record
from arancio.sessions.manager import SessionManager
from arancio.sessions.registry import SessionRegistryEntry
from arancio.sessions.session import Session, SessionConfiguration
from arancio.storage.manager import StorageManager


def _configuration() -> SessionConfiguration:
    """Build a configuration covering every permission category.

    Returns:
        A complete session configuration.
    """
    return SessionConfiguration(
        provider="openai",
        model_name="gpt-5",
        thinking_effort="high",
        permissions={
            PermissionCategory.READ: PermissionLevel.AUTO,
            PermissionCategory.WRITE: PermissionLevel.ASK,
            PermissionCategory.WEB: PermissionLevel.NONE,
            PermissionCategory.EXECUTE: PermissionLevel.ASK,
            PermissionCategory.PLUGIN: PermissionLevel.ASK,
        },
    )


def _open_log(manager: SessionManager, working_directory: Path) -> Session:
    """Create a session and put its log on disk.

    A new session keeps its creation header in memory until the first write
    that needs saving, so a test reading the log has to ask for that write.

    Args:
        manager: the manager opening the chat.
        working_directory: the directory the chat runs in.

    Returns:
        The new session, its header already written.
    """
    session = manager.create(
        working_directory=working_directory, configuration=_configuration()
    )
    manager.session_recorder.flush()
    return session


def _reopen(session: Session, storage_manager: StorageManager) -> SessionManager:
    """Open a second manager over the same root and load a session from disk.

    Args:
        session: the session to load back.
        storage_manager: the storage manager holding the sessions root.

    Returns:
        A manager whose current session was read from the log.
    """
    manager = SessionManager(
        storage_manager,
        _configuration(),
        root=storage_manager.root,
        tool_session=ToolSession(),
    )
    manager.load(session.id)
    return manager


def test_every_finalized_message_survives_a_round_trip() -> None:
    """A restored session must hand the model exactly what it said before."""
    messages = [
        UserMessage(content="raw", display_text="shown"),
        AssistantMessage(content="answer"),
        ReasoningMessage(content="thought", item={"type": "reasoning", "id": "r1"}),
        ToolCallMessage(
            content="ReadFileTool({})",
            id="call_1",
            name="ReadFileTool",
            arguments={"file_path": "/tmp/a"},
        ),
        ToolResultMessage(content={"file_path": "/tmp/a"}, id="call_1"),
        ToolErrorMessage(content="failed", id="call_1"),
    ]

    records = [message_to_record(message) for message in messages]

    assert [message_from_record(record) for record in records] == messages


def test_an_error_round_trips_as_shown_but_not_remembered() -> None:
    """The flags travel with the record, so replay routes it where it was."""
    error = ErrorMessage(content="Max turns exceeded")

    record = message_to_record(error)

    assert record["type"] == "error"
    assert record["in_history"] is False
    assert record["visible"] is True
    assert message_from_record(record) == error


def test_a_streaming_fragment_has_no_record_at_all() -> None:
    """The schema has no type for a chunk, so the codec refuses to invent one.

    This is load-bearing rather than defensive: an ``AssistantChunkMessage``
    also passes ``isinstance(m, AssistantMessage)``, so without this check the
    branch below would save every fragment as a finished reply.
    """
    assert message_to_record(AssistantChunkMessage(content="frag")) is None


def test_a_message_type_the_schema_cannot_hold_is_refused() -> None:
    """A warning is shown at startup and never belongs to a conversation."""
    with pytest.raises(ValueError):
        message_to_record(WarningMessage(content="heads up"))


def test_a_session_exists_from_startup_but_its_log_does_not(
    session_manager: SessionManager, storage_manager: StorageManager
) -> None:
    """Opening arancio and closing it again must leave nothing behind."""
    session = session_manager.current

    assert session.created_on_disk is False
    assert not session.path.exists()
    assert list(storage_manager.root.rglob("*.jsonl")) == []
    assert session_manager.registry.contains(session.id) is False


def test_the_first_write_puts_the_header_down_ahead_of_itself(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """The header waits in memory, so the first real write carries it to disk."""
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )

    session_manager.session_recorder.message(UserMessage(content="hello"))

    records = [json.loads(line) for line in session.path.read_text().splitlines()]
    assert [record["type"] for record in records] == ["session_created", "user"]
    assert session_manager.registry.contains(session.id) is True


def test_a_new_session_opens_a_dated_log_with_a_versioned_header(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """The first line states the schema version a later read validates."""
    session = _open_log(session_manager, tmp_path)

    assert session.path.suffix == ".jsonl"
    assert session.path.stem == session.id
    assert session.path.parent.parent.parent.parent.parent.name == "sessions"
    header = json.loads(session.path.read_text().splitlines()[0])
    assert header["type"] == "session_created"
    assert header["format_version"] == 1
    assert header["id"] == session.id
    assert header["configuration"] == _configuration().to_dict()


def test_forked_from_is_none_for_an_ordinary_session(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """A session created or resumed normally carries no fork provenance."""
    session = _open_log(session_manager, tmp_path)

    assert session.forked_from is None
    header = json.loads(session.path.read_text().splitlines()[0])
    assert header["forked_from"] is None


def test_starting_a_new_chat_forgets_the_files_the_old_one_read(
    storage_manager: StorageManager, tmp_path: Path
) -> None:
    """The read-first guard is per chat: a new one may not overwrite blindly."""
    tool_session = ToolSession()
    manager = SessionManager(
        storage_manager,
        _configuration(),
        root=storage_manager.root,
        tool_session=tool_session,
    )
    _open_log(manager, tmp_path)
    tool_session.record_read("/tmp/file.txt", 1.0)

    session = manager.discard_and_create(
        working_directory=tmp_path, configuration=_configuration()
    )

    assert manager.current is session
    assert tool_session.is_known("/tmp/file.txt") is False


def test_the_model_and_the_user_are_replayed_from_the_same_log(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """One log, two views: the flags on each record decide which view it joins.

    The ``!`` attribution line is the case that needs both walks to disagree — the model
    must see it, the user must not.
    """
    session = _open_log(session_manager, tmp_path)
    recorder = session_manager.session_recorder
    recorder.message(UserMessage(content="hello"))
    recorder.message(
        UserMessage(content="User explicitly ran the following command:"),
        visible=False,
    )
    recorder.command(raw_input="/model gpt-5", name="model", args=["gpt-5"])
    recorder.message(ErrorMessage(content="something went wrong"))

    reopened = _reopen(session, storage_manager)

    assert [m.content for m in reopened.model_history()] == [
        "hello",
        "User explicitly ran the following command:",
    ]
    assert [m.content for m in reopened.visible_messages()] == [
        "hello",
        "/model gpt-5",
        "something went wrong",
    ]


def test_a_command_line_is_rebuilt_from_its_own_record(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """A command has no message record, so the typed line is reconstructed."""
    session = _open_log(session_manager, tmp_path)
    session_manager.session_recorder.command(
        raw_input="/effort high", name="effort", args=["high"]
    )

    replayed = _reopen(session, storage_manager).visible_messages()

    assert replayed == [UserMessage(content="/effort high")]


def test_loading_restores_the_configuration_the_session_ended_with(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """Later configuration records supersede the header, not the global defaults."""
    session = _open_log(session_manager, tmp_path)
    changed = SessionConfiguration(
        provider="anthropic",
        model_name="claude",
        thinking_effort=None,
        permissions={category: PermissionLevel.AUTO for category in PermissionCategory},
    )
    session.configuration = changed
    session_manager.session_recorder.state_changed()

    restored = _reopen(session, storage_manager).current

    assert restored.configuration == changed


def test_reading_picks_the_last_state_changed_across_intervening_messages(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """A message written between two state snapshots must not hide the later one."""
    session = _open_log(session_manager, tmp_path)
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    second = SessionConfiguration(
        provider="anthropic",
        model_name="claude",
        thinking_effort=None,
        permissions={category: PermissionLevel.AUTO for category in PermissionCategory},
    )
    session.configuration = _configuration()
    session.working_directory = first_directory.resolve()
    session.name = "first-name"
    session_manager.session_recorder.state_changed()
    session_manager.session_recorder.message(UserMessage(content="in between"))
    session.configuration = second
    session.working_directory = second_directory.resolve()
    session.name = "second-name"
    session_manager.session_recorder.state_changed()

    restored = _reopen(session, storage_manager).current

    assert restored.configuration == second
    assert restored.working_directory == second_directory.resolve()
    assert restored.name == "second-name"


def test_scanning_recovers_the_same_latest_state_without_a_full_parse(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """The registry must not show a session's stale, creation-time state."""
    session = _open_log(session_manager, tmp_path)
    moved = tmp_path / "moved"
    changed = SessionConfiguration(
        provider="anthropic",
        model_name="claude",
        thinking_effort=None,
        permissions={category: PermissionLevel.AUTO for category in PermissionCategory},
    )
    session.configuration = changed
    session.working_directory = moved.resolve()
    session.name = "renamed"
    session_manager.session_recorder.state_changed()

    reopened = SessionManager(
        storage_manager, _configuration(), root=storage_manager.root
    )
    entry = reopened.registry.get(session.id)

    assert entry.working_directory == moved.resolve()
    assert entry.configuration == changed
    assert entry.name == "renamed"


def test_a_malformed_earlier_state_changed_only_fails_a_full_load(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """The scan only trusts the winning record; a full load checks every one."""
    session = _open_log(session_manager, tmp_path)
    malformed = json.dumps(
        {
            "type": "state_changed",
            "timestamp": "2026-01-01T00:00:00Z",
            "working_directory": str(tmp_path),
            "configuration": "not-a-mapping",
            "name": session.id,
        }
    )
    with session.path.open("a") as handle:
        handle.write(malformed + "\n")
    session_manager.session_recorder.state_changed()

    reopened = SessionManager(
        storage_manager, _configuration(), root=storage_manager.root
    )
    entry = reopened.registry.get(session.id)
    assert entry.configuration == _configuration()

    with pytest.raises(ValueError):
        reopened.load(session.id)


def test_loading_answers_a_tool_call_that_never_got_a_result(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """A crash mid-call must not leave the model waiting on an answer forever."""
    session = _open_log(session_manager, tmp_path)
    session_manager.session_recorder.message(
        ToolCallMessage(
            content="", id="call_1", name="ReadFileTool", arguments={"file_path": "/a"}
        )
    )

    history = _reopen(session, storage_manager).model_history()

    closing = history[-1]
    assert isinstance(closing, ToolErrorMessage)
    assert closing.id == "call_1"
    assert "may or may not have completed" in closing.content


def test_the_read_guard_is_restored_only_for_unchanged_files(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """Re-arming a stale read would let the model overwrite someone else's edit."""
    unchanged = tmp_path / "same.txt"
    unchanged.write_text("a")
    changed = tmp_path / "moved.txt"
    changed.write_text("a")
    session = _open_log(session_manager, tmp_path)
    recorder = session_manager.session_recorder
    recorder.file_state(str(unchanged), unchanged.stat().st_mtime)
    recorder.file_state(str(changed), changed.stat().st_mtime)
    changed.write_text("edited elsewhere")

    tool_session = ToolSession()
    reopened = SessionManager(
        storage_manager,
        _configuration(),
        root=storage_manager.root,
        tool_session=tool_session,
    )
    reopened.load(session.id)
    reopened.restore_file_states()

    assert tool_session.is_known(str(unchanged)) is True
    assert tool_session.is_known(str(changed)) is False


def test_a_working_directory_that_vanished_falls_back_and_says_so(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """Resuming into a deleted directory has to degrade, not fail."""
    gone = tmp_path / "gone"
    gone.mkdir()
    session = _open_log(session_manager, gone)
    gone.rmdir()

    reopened = _reopen(session, storage_manager)
    directory, notice = reopened.restored_working_directory(tmp_path)

    assert directory == tmp_path
    assert notice is not None
    assert "no longer exists" in notice


@pytest.mark.parametrize(
    ("lines", "reason"),
    [
        ([], "Missing session_created record"),
        (['{"type": "user", "content": "x"}'], "Missing session_created record"),
        (["not json at all"], "Invalid JSON at line 1"),
    ],
    ids=["empty", "no-header", "unparseable"],
)
def test_a_log_without_a_valid_header_is_damaged(
    storage_manager: StorageManager, lines: list[str], reason: str
) -> None:
    """A session that cannot be trusted is listed, not loaded."""
    log = storage_manager.root / "sessions" / "cwd" / "2026" / "01" / "01" / "abc.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text("\n".join(lines))

    manager = SessionManager(
        storage_manager, _configuration(), root=storage_manager.root
    )
    entry = manager.registry.get("abc")

    assert entry.status == "damaged"
    assert entry.damage_reason == reason


def test_a_truncated_last_line_damages_the_session(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """A process killed mid-write leaves half a record, and the scan spots it."""
    session = _open_log(session_manager, tmp_path)
    with session.path.open("a") as handle:
        handle.write('{"type": "user", "conte')

    manager = SessionManager(
        storage_manager, _configuration(), root=storage_manager.root
    )

    assert manager.registry.get(session.id).status == "damaged"
    with pytest.raises(ValueError):
        manager.load(session.id)


def test_the_same_id_in_two_files_damages_both(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """Neither copy can be trusted to be the real chat, so neither is offered."""
    session = _open_log(session_manager, tmp_path)
    duplicate = session.path.parent.parent / session.path.name
    duplicate.write_text(session.path.read_text())

    manager = SessionManager(
        storage_manager, _configuration(), root=storage_manager.root
    )
    damaged = [e for e in manager.registry.entries if e.id == session.id]

    assert len(damaged) == 2
    assert all(entry.status == "damaged" for entry in damaged)
    assert all("Duplicate session ID" in entry.damage_reason for entry in damaged)


def test_find_resolves_an_exact_id_over_a_name_match(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """A query that happens to also be a name fragment must not create ambiguity."""
    session = _open_log(session_manager, tmp_path)
    session_manager.registry.entries.append(
        SessionRegistryEntry(
            id="other",
            explicit_name=f"prefix-{session.id}-suffix",
            working_directory=tmp_path,
            configuration=_configuration(),
            log_path=tmp_path / "other.jsonl",
            status="healthy",
        )
    )

    matches = session_manager.registry.find(session.id)

    assert [entry.id for entry in matches] == [session.id]


def test_find_matches_a_name_fragment_case_insensitively(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """The registry has no other way to look a session up by hand."""
    session = _open_log(session_manager, tmp_path)

    matches = session_manager.registry.find(session.id[:8].upper())

    assert [entry.id for entry in matches] == [session.id]


def test_find_returns_both_copies_of_a_duplicated_id(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """A query for a colliding ID must not silently pick one copy."""
    session = _open_log(session_manager, tmp_path)
    duplicate = session.path.parent.parent / session.path.name
    duplicate.write_text(session.path.read_text())
    manager = SessionManager(
        storage_manager, _configuration(), root=storage_manager.root
    )

    matches = manager.registry.find(session.id)

    assert len(matches) == 2


def test_find_returns_nothing_for_an_unknown_query(
    session_manager: SessionManager,
) -> None:
    """No match is how the command knows to report an error, not resume nothing."""
    assert session_manager.registry.find("nosuchsession") == []


def test_loading_an_unknown_session_is_refused(
    session_manager: SessionManager,
) -> None:
    """A stale id from a deleted file is a clear error, not an empty chat."""
    with pytest.raises(ValueError):
        session_manager.load("nosuchsession")


def test_a_clean_flush_writes_a_checksum_covering_the_log(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """The checksum says the file is exactly what the last flush left."""
    session = _open_log(session_manager, tmp_path)

    assert SessionChecksum.path(session.path).exists()
    assert SessionChecksum.verify(session.path) is True


def test_a_missing_checksum_lists_the_session_as_unverified_and_load_heals_it(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """No checksum means "not checked", and a clean load proves the log good."""
    session = _open_log(session_manager, tmp_path)
    SessionChecksum.path(session.path).unlink()

    manager = SessionManager(
        storage_manager, _configuration(), root=storage_manager.root
    )
    entry = manager.registry.get(session.id)

    assert entry.status == "unverified"
    assert not SessionChecksum.path(session.path).exists()

    manager.load(session.id)

    assert entry.status == "healthy"
    assert SessionChecksum.verify(session.path) is True


def test_a_stale_checksum_marks_the_session_damaged_but_load_heals_it(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """A checksum write that died mid-flush must not lock out a healthy log."""
    session = _open_log(session_manager, tmp_path)
    SessionChecksum.path(session.path).write_text("0" * 64)

    manager = SessionManager(
        storage_manager, _configuration(), root=storage_manager.root
    )
    entry = manager.registry.get(session.id)

    assert entry.status == "damaged"
    assert "checksum" in entry.damage_reason

    manager.load(session.id)

    assert entry.status == "healthy"
    assert SessionChecksum.verify(session.path) is True
    reopened = SessionManager(
        storage_manager, _configuration(), root=storage_manager.root
    )
    assert reopened.registry.get(session.id).status == "healthy"


def test_an_orphan_checksum_file_is_ignored(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """Checksums of logs that no longer exist must not surface as sessions."""
    session = _open_log(session_manager, tmp_path)
    orphan = session.path.parent / "nosuchsession.jsonl.sha256"
    orphan.write_text("0" * 64)

    manager = SessionManager(
        storage_manager, _configuration(), root=storage_manager.root
    )

    assert all(
        entry.log_path.name != "nosuchsession.jsonl"
        for entry in manager.registry.entries
    )


@pytest.mark.parametrize("explicit_name", [None, "demo", ""])
def test_session_name_fallback_survives_round_trip(
    session_manager: SessionManager,
    storage_manager: StorageManager,
    tmp_path: Path,
    explicit_name: str | None,
) -> None:
    """Only explicit names are stored; display names fall back without mutation."""
    session = _open_log(session_manager, tmp_path)
    assert session.explicit_name is None
    assert session.name == session.id
    assert session.explicit_name is None
    assert json.loads(session.path.read_text().splitlines()[0])["name"] is None
    session.name = explicit_name
    session_manager.session_recorder.state_changed()
    assert (
        json.loads(session.path.read_text().splitlines()[-1])["name"] == explicit_name
    )
    reopened = _reopen(session, storage_manager)
    restored = reopened.current
    expected = session.id if explicit_name is None else explicit_name
    assert restored.explicit_name == explicit_name
    assert restored.name == expected
    for manager in (session_manager, reopened):
        entry = manager.registry.get(session.id)
        assert entry.explicit_name == explicit_name
        assert entry.name == expected
        assert manager.registry.find(session.id) == [entry]
        assert entry in manager.registry.find(expected[:8])


def test_clearing_session_name_restores_id_fallback(
    session_manager: SessionManager,
    storage_manager: StorageManager,
    tmp_path: Path,
) -> None:
    """Clearing a previous name persists null and restores the ID label."""
    session = _open_log(session_manager, tmp_path)
    session.name = "demo"
    session_manager.session_recorder.state_changed()
    session.name = None
    session_manager.session_recorder.state_changed()
    restored = _reopen(session, storage_manager).current
    assert restored.explicit_name is None
    assert restored.name == session.id


@pytest.mark.parametrize("invalid_name", [123, False, [], {}, "missing"])
@pytest.mark.parametrize("record_type", ["session_created", "state_changed"])
def test_invalid_session_names_are_rejected(
    session_manager: SessionManager,
    storage_manager: StorageManager,
    tmp_path: Path,
    invalid_name: object,
    record_type: str,
) -> None:
    """Both state record types require a name containing a string or null."""
    from arancio.sessions.validator import SessionValidator

    session = _open_log(session_manager, tmp_path)
    if record_type == "state_changed":
        session_manager.session_recorder.state_changed()
    records = [json.loads(line) for line in session.path.read_text().splitlines()]
    if invalid_name == "missing":
        del records[-1]["name"]
    else:
        records[-1]["name"] = invalid_name
    session.path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    validator = SessionValidator(storage_manager)
    assert validator.scan(session.path).status == "damaged"
    with pytest.raises(ValueError, match="name is missing or invalid"):
        validator.read(session.path)


@pytest.mark.parametrize("invalid_forked_from", [123, False, [], {}, "missing"])
def test_invalid_forked_from_headers_are_rejected(
    session_manager: SessionManager,
    storage_manager: StorageManager,
    tmp_path: Path,
    invalid_forked_from: object,
) -> None:
    """The header must carry a ``forked_from`` containing a string or null."""
    from arancio.sessions.validator import SessionValidator

    session = _open_log(session_manager, tmp_path)
    records = [json.loads(line) for line in session.path.read_text().splitlines()]
    if invalid_forked_from == "missing":
        del records[0]["forked_from"]
    else:
        records[0]["forked_from"] = invalid_forked_from
    session.path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    validator = SessionValidator(storage_manager)
    with pytest.raises(ValueError, match="forked_from is missing or invalid"):
        validator.read(session.path)
