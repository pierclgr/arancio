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
        },
    )


def _reopen(session: Session, storage_manager: StorageManager) -> SessionManager:
    """Open a second manager over the same root and load a session from disk.

    Args:
        session: the session to load back.
        storage_manager: the storage manager holding the sessions root.

    Returns:
        A manager whose current session was read from the log.
    """
    manager = SessionManager(
        storage_manager, root=storage_manager.root, tool_session=ToolSession()
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


def test_a_new_session_opens_a_dated_log_with_a_versioned_header(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """The first line states the schema version a later read validates."""
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )

    assert session.path.suffix == ".jsonl"
    assert session.path.stem == session.id
    assert session.path.parent.parent.parent.parent.parent.name == "sessions"
    header = json.loads(session.path.read_text().splitlines()[0])
    assert header["type"] == "session_created"
    assert header["format_version"] == 1
    assert header["id"] == session.id
    assert header["configuration"] == _configuration().to_dict()


def test_starting_a_new_chat_forgets_the_files_the_old_one_read(
    storage_manager: StorageManager, tmp_path: Path
) -> None:
    """The read-first guard is per chat: a new one may not overwrite blindly."""
    tool_session = ToolSession()
    manager = SessionManager(
        storage_manager, root=storage_manager.root, tool_session=tool_session
    )
    manager.create(working_directory=tmp_path, configuration=_configuration())
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
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )
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
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )
    session_manager.session_recorder.command(
        raw_input="/effort high", name="effort", args=["high"]
    )

    replayed = _reopen(session, storage_manager).visible_messages()

    assert replayed == [UserMessage(content="/effort high")]


def test_loading_restores_the_configuration_the_session_ended_with(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """Later configuration records supersede the header, not the global defaults."""
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )
    changed = SessionConfiguration(
        provider="anthropic",
        model_name="claude",
        thinking_effort=None,
        permissions={category: PermissionLevel.AUTO for category in PermissionCategory},
    )
    session_manager.session_recorder.configuration(changed)

    restored = _reopen(session, storage_manager).require_current()

    assert restored.configuration == changed


def test_loading_answers_a_tool_call_that_never_got_a_result(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """A crash mid-call must not leave the model waiting on an answer forever."""
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )
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
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )
    recorder = session_manager.session_recorder
    recorder.file_state(str(unchanged), unchanged.stat().st_mtime)
    recorder.file_state(str(changed), changed.stat().st_mtime)
    changed.write_text("edited elsewhere")

    tool_session = ToolSession()
    reopened = SessionManager(
        storage_manager, root=storage_manager.root, tool_session=tool_session
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
    session = session_manager.create(
        working_directory=gone, configuration=_configuration()
    )
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

    manager = SessionManager(storage_manager, root=storage_manager.root)
    entry = manager.registry.get("abc")

    assert entry.status == "damaged"
    assert entry.damage_reason == reason


def test_a_truncated_last_line_damages_the_session(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """A process killed mid-write leaves half a record, and the scan spots it."""
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )
    with session.path.open("a") as handle:
        handle.write('{"type": "user", "conte')

    manager = SessionManager(storage_manager, root=storage_manager.root)

    assert manager.registry.get(session.id).status == "damaged"
    with pytest.raises(ValueError):
        manager.load(session.id)


def test_the_same_id_in_two_files_damages_both(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """Neither copy can be trusted to be the real chat, so neither is offered."""
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )
    duplicate = session.path.parent.parent / session.path.name
    duplicate.write_text(session.path.read_text())

    manager = SessionManager(storage_manager, root=storage_manager.root)
    damaged = [e for e in manager.registry.entries if e.id == session.id]

    assert len(damaged) == 2
    assert all(entry.status == "damaged" for entry in damaged)
    assert all("Duplicate session ID" in entry.damage_reason for entry in damaged)


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
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )

    assert SessionChecksum.path(session.path).exists()
    assert SessionChecksum.verify(session.path) is True


def test_a_missing_checksum_lists_the_session_as_unverified_and_load_heals_it(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """No checksum means "not checked", and a clean load proves the log good."""
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )
    SessionChecksum.path(session.path).unlink()

    manager = SessionManager(storage_manager, root=storage_manager.root)
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
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )
    SessionChecksum.path(session.path).write_text("0" * 64)

    manager = SessionManager(storage_manager, root=storage_manager.root)
    entry = manager.registry.get(session.id)

    assert entry.status == "damaged"
    assert "checksum" in entry.damage_reason

    manager.load(session.id)

    assert entry.status == "healthy"
    assert SessionChecksum.verify(session.path) is True
    reopened = SessionManager(storage_manager, root=storage_manager.root)
    assert reopened.registry.get(session.id).status == "healthy"


def test_an_orphan_checksum_file_is_ignored(
    session_manager: SessionManager, storage_manager: StorageManager, tmp_path: Path
) -> None:
    """Checksums of logs that no longer exist must not surface as sessions."""
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )
    orphan = session.path.parent / "nosuchsession.jsonl.sha256"
    orphan.write_text("0" * 64)

    manager = SessionManager(storage_manager, root=storage_manager.root)

    assert all(
        entry.log_path.name != "nosuchsession.jsonl"
        for entry in manager.registry.entries
    )
