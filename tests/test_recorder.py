"""Tests for the write side of a chat session.

``SessionRecorder`` is the only class that builds a session record or touches the log.
The part worth the most attention is what happens when a write fails: the event stays
unsaved, the half-written tail is remembered, and the next attempt rolls it back before
retrying.
"""

import json
from pathlib import Path
from typing import Iterator, List

import pytest

from arancio.core.messages import (
    AssistantChunkMessage,
    AssistantMessage,
    ErrorMessage,
    Message,
    UserMessage,
)
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.core.tools.session import ToolSession, shared_session
from arancio.sessions.manager import SessionManager
from arancio.sessions.session import SessionConfiguration
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
        permissions={c: PermissionLevel.ASK for c in PermissionCategory},
    )


def _records(path: Path) -> List[dict]:
    """Read back every record written to a session log.

    Args:
        path: the session log to read.

    Returns:
        One decoded record per line.
    """
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class _FlakyStorage(StorageManager):
    """Storage that can be told to fail the next append or file creation.

    A failing append writes a partial line first, which is what makes the
    recovery offset meaningful: the tail on disk is real but unconfirmed.

    Attributes:
        fail_append: whether the next ``append_line`` should fail.
        create_error: an exception the next ``create_file`` should raise.
    """

    def __init__(self, root: Path) -> None:
        """Initialize the storage with no failures armed.

        Args:
            root: the arancio root directory.
        """
        super().__init__(root=root)
        self.fail_append = False
        self.create_error: Exception | None = None

    def append_line(self, path: Path, line: str) -> None:
        """Append a line, or write a partial one and fail when armed.

        Args:
            path: the file to append to.
            line: the record to append.

        Raises:
            OSError: when a failure is armed, after a partial write.
        """
        if self.fail_append:
            self.fail_append = False
            with path.open("ab") as handle:
                handle.write(line.encode("utf-8")[:8])
            raise OSError("no space left on device")
        StorageManager.append_line(path, line)

    def create_file(self, path: Path, line: str | None = None) -> None:
        """Create a file, or fail with the armed exception.

        An armed ``create_error`` is raised instead of creating the file.

        Args:
            path: the file to create.
            line: the first line to write.
        """
        if self.create_error is not None:
            error, self.create_error = self.create_error, None
            raise error
        StorageManager.create_file(path, line)


@pytest.fixture
def flaky(tmp_path: Path) -> _FlakyStorage:
    """Return a storage manager whose writes can be made to fail.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A storage manager with no failures armed yet.
    """
    return _FlakyStorage(root=tmp_path / "arancio")


def _open_session(
    storage: StorageManager, tmp_path: Path
) -> tuple[SessionManager, Path]:
    """Open a manager with a fresh chat.

    Args:
        storage: the storage manager backing the log.
        tmp_path: the working directory to record.

    Returns:
        The manager and the open session's log path.
    """
    manager = SessionManager(storage, root=storage.root, tool_session=ToolSession())
    session = manager.create(working_directory=tmp_path, configuration=_configuration())
    return manager, session.path


def test_a_command_is_saved_as_shown_but_not_remembered(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """A slash command is replayed to the user and hidden from the model.

    The reader enforces exactly these two values, so a command written any other way
    makes its own session unloadable.
    """
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )

    assert (
        session_manager.session_recorder.command(
            raw_input="/model gpt-5", name="model", args=["gpt-5"]
        )
        is None
    )

    record = _records(session.path)[-1]
    assert record["type"] == "command"
    assert record["raw_input"] == "/model gpt-5"
    assert record["args"] == ["gpt-5"]
    assert record["in_history"] is False
    assert record["visible"] is True


def test_a_streaming_fragment_is_dropped_before_it_reaches_the_log(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """Chunks arrive constantly, so the recorder takes them and writes nothing."""
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )
    before = len(_records(session.path))

    assert (
        session_manager.session_recorder.message(AssistantChunkMessage(content="frag"))
        is None
    )

    assert len(_records(session.path)) == before


def test_an_error_is_saved_like_anything_else(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """A mistyped command whose only output was an error still leaves a trace."""
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )

    session_manager.session_recorder.message(ErrorMessage(content="boom"))

    record = _records(session.path)[-1]
    assert record["type"] == "error"
    assert record["in_history"] is False
    assert record["visible"] is True


def test_a_hidden_message_is_saved_out_of_the_replayed_log(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """Visibility is the caller's to state, since core owns no UI."""
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )

    session_manager.session_recorder.message(
        UserMessage(content="User explicitly ran the following command:"),
        visible=False,
    )

    record = _records(session.path)[-1]
    assert record["visible"] is False
    assert record["in_history"] is True


def test_reads_picked_up_by_the_guard_are_persisted_with_the_next_write(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """Core never announces a read, so the session diffs the guard itself."""
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )
    shared_session.record_read("/tmp/read.txt", 123.0)

    session_manager.session_recorder.message(AssistantMessage(content="done"))

    types = [record["type"] for record in _records(session.path)]
    assert types[-1] == "file_state_changed"
    assert session.file_states["/tmp/read.txt"] == 123.0


def test_a_stream_is_passed_through_before_it_is_recorded(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """The UI should not wait on a disk write to render the next message."""
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )
    seen_counts: List[int] = []

    def _stream() -> Iterator[Message]:
        """Yield two messages, noting how much was saved between them.

        Yields:
            Two assistant messages.
        """
        yield AssistantMessage(content="one")
        seen_counts.append(len(_records(session.path)))
        yield AssistantMessage(content="two")

    produced = list(session_manager.session_recorder.record_stream(_stream()))

    assert [m.content for m in produced] == ["one", "two"]
    assert seen_counts == [2]


def test_a_failed_write_is_rolled_back_and_retried(
    flaky: _FlakyStorage, tmp_path: Path
) -> None:
    """The partial tail is remembered, then truncated away before the retry.

    Without the offset the retry would append after the half-written record and corrupt
    the log permanently.
    """
    manager, log = _open_session(flaky, tmp_path)
    session = manager.require_current()
    confirmed = flaky.file_size(log)
    flaky.fail_append = True

    error = manager.session_recorder.message(UserMessage(content="hello"))

    assert isinstance(error, ErrorMessage)
    assert error.content.startswith("Could not save session:")
    assert session.recovery_offset == confirmed
    assert flaky.file_size(log) > confirmed

    assert manager.session_recorder.flush() is None

    assert session.recovery_offset is None
    records = _records(log)
    assert len(records) == 2
    assert records[1]["content"] == "hello"


def test_a_session_whose_first_write_failed_leaves_nothing_behind(
    flaky: _FlakyStorage, tmp_path: Path
) -> None:
    """A log that never got its header is removed, not left as a stub.

    The session object still exists in memory, so the chat keeps working; it simply is
    not on disk or in the registry yet.
    """
    flaky.create_error = OSError("read-only file system")
    manager = SessionManager(flaky, root=flaky.root, tool_session=ToolSession())

    session = manager.create(working_directory=tmp_path, configuration=_configuration())

    assert session.created_on_disk is False
    assert session.recovery_offset is None
    assert not session.path.exists()
    assert manager.registry.contains(session.id) is False


def test_a_colliding_log_name_records_no_recovery_offset(
    flaky: _FlakyStorage, tmp_path: Path
) -> None:
    """Nothing was written, so there is no unconfirmed tail to roll back."""
    flaky.create_error = FileExistsError("already there")
    manager = SessionManager(flaky, root=flaky.root, tool_session=ToolSession())

    session = manager.create(working_directory=tmp_path, configuration=_configuration())

    assert session.recovery_offset is None
    assert session.created_on_disk is False


def test_a_successful_write_registers_the_session(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """``flush`` is the single funnel, so it is what keeps the registry true."""
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )

    assert session_manager.registry.contains(session.id) is True


def test_a_configuration_change_updates_the_session_and_the_log(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """The in-memory session and its log must not drift apart."""
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )
    changed = SessionConfiguration(
        provider="anthropic",
        model_name="claude",
        thinking_effort=None,
        permissions={c: PermissionLevel.AUTO for c in PermissionCategory},
    )

    session_manager.session_recorder.state_changed(changed, session.working_directory)

    assert session.configuration == changed
    assert _records(session.path)[-1]["type"] == "state_changed"


def test_a_directory_change_is_resolved_before_it_is_stored(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    """A relative or symlinked path would not survive a restart."""
    session = session_manager.create(
        working_directory=tmp_path, configuration=_configuration()
    )
    nested = tmp_path / "sub"
    nested.mkdir()

    session_manager.session_recorder.state_changed(session.configuration, nested)

    assert session.working_directory == nested.resolve()
    assert _records(session.path)[-1]["type"] == "state_changed"
