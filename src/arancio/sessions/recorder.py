"""Write side of a chat session: turning facts into durable events."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from arancio.core.messages import (
    ErrorMessage,
    Message,
    ToolCallMessage,
    ToolErrorMessage,
)
from arancio.core.tools.session import ToolSession
from arancio.sessions.checksum import SessionChecksum
from arancio.sessions.codec import (
    is_message_record,
    message_from_record,
    message_to_record,
)
from arancio.sessions.constants import SESSION_FORMAT_VERSION
from arancio.sessions.session import Session
from arancio.storage.manager import StorageManager

if TYPE_CHECKING:
    from arancio.sessions.manager import SessionManager


class SessionRecorder:
    """Apply what the app did to a session and persist it.

    Owns the whole write path: building each event record, applying the state change
    it describes, appending it to the session and flushing it to the JSONL log, with
    the retry bookkeeping a failed write needs, and keeping the registry true
    afterwards. It asks its :class:`SessionManager` which chat is open rather than
    taking one per call, so a caller only passes what it wants written.

    Attributes:
        _storage_manager: the file-only storage service writing the session log.
        _session_manager: the manager owning the open chat and the session registry.
        _tool_session: the shared read guard whose state the session persists.
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        session_manager: SessionManager,
        tool_session: ToolSession,
    ) -> None:
        """Initialize the recorder with the managers it works through.

        The manager is still being built when it hands itself over, so the
        reference is only stored here and first read on the earliest write.

        Args:
            storage_manager: the file-only storage service for the arancio directory.
            session_manager: the manager owning the open chat and the session registry.
            tool_session: the read guard the file tools record into, whose state a
                resumed chat needs restored.
        """
        self._storage_manager: StorageManager = storage_manager
        self._session_manager: SessionManager = session_manager
        self._tool_session: ToolSession = tool_session

    def _session(self) -> Session:
        """Return the chat every write goes into.

        Returns:
            The manager's open session.
        """
        return self._session_manager.current

    def created(self) -> None:
        """Add the creation header that opens a session log, without saving it.

        The header is the one record that is not flushed when it is built: an
        untouched session must leave no file behind. It stays unsaved until the
        first write that has something to save, which :meth:`flush` then writes
        ahead of it, in order.

        The header also records the session's fork provenance, ``None``
        unless the session was created by ``SessionManager.fork``.
        """
        session = self._session()
        session.add_event(
            {
                "type": "session_created",
                "format_version": SESSION_FORMAT_VERSION,
                "timestamp": session.created_at.isoformat(),
                "id": session.id,
                "name": session.explicit_name,
                "creation_working_directory": str(session.creation_working_directory),
                "working_directory": str(session.working_directory),
                "configuration": session.configuration.to_dict(),
                "forked_from": session.forked_from,
            }
        )

    def message(self, message: Message, visible: bool = True) -> ErrorMessage | None:
        """Record one finalized message, skipping what is not meant to be saved.

        The codec has no record for a streaming chunk and returns ``None`` for one,
        so a caller can hand over everything it produces unfiltered. Every other
        message becomes a message record carrying its own routing flags, errors
        included: the session replays what the user saw and what the model saw. Any
        file read the guard picked up meanwhile is persisted with it.

        Args:
            message: the message to persist.
            visible: whether the record belongs in the replayed UI log. A caller
                passes ``False`` for a message the user was never shown.

        Returns:
            The temporary persistence error notice, or ``None`` when the message was
            saved or is not meant to be.
        """
        record = message_to_record(message, visible)
        if record is None:
            return None
        error = self.event(record)
        return error or self._sync_file_states()

    def _sync_file_states(self) -> ErrorMessage | None:
        """Record every read the guard holds that the session has not saved yet.

        The file tools record into the shared guard as they run, so comparing it
        with what the session already stored is enough to persist the read-first
        state without core announcing it.

        Returns:
            The first persistence error notice, or ``None`` when every new read
            was saved or none was pending.
        """
        session = self._session()
        error: ErrorMessage | None = None
        for path, mtime in self._tool_session.snapshot().items():
            if session.file_states.get(path) == mtime:
                continue
            error = error or self.file_state(path, mtime)
        return error

    def record_stream(self, messages: Iterator[Message]) -> Iterator[Message]:
        """Record each message as it passes, injecting any persistence failure.

        Every message is yielded through unchanged, so the caller keeps the whole
        stream and decides on its own what to render.

        Args:
            messages: the message stream to record as it is consumed.

        Yields:
            Each message in order, each followed by an :class:`ErrorMessage` when
            saving that message failed.
        """
        for message in messages:
            yield message
            error = self.message(message)
            if error:
                yield error

    def state_changed(self) -> ErrorMessage | None:
        """Record the session's current configuration, working directory and name.

        The caller is responsible for updating those fields on the session before
        calling this; the recorder only builds the record and persists it. All
        three are always recorded together, so the log carries one atomic record
        of "the state at this point" rather than several independently-timed
        ones.

        Returns:
            The temporary persistence error notice, or ``None`` on success.
        """
        session = self._session()
        return self.event(
            {
                "type": "state_changed",
                "timestamp": self._utc_timestamp(),
                "configuration": session.configuration.to_dict(),
                "working_directory": str(session.working_directory),
                "name": session.explicit_name,
            }
        )

    def file_state(self, path: str, mtime: float) -> ErrorMessage | None:
        """Record one file-read safety state without exposing it to the model.

        Args:
            path: the canonical absolute file path.
            mtime: the file modification time observed after a successful tool call.

        Returns:
            The temporary persistence error notice, or ``None`` on success.
        """
        session = self._session()
        session.file_states[path] = mtime
        return self.event(
            {
                "type": "file_state_changed",
                "timestamp": self._utc_timestamp(),
                "file_path": path,
                "mtime": mtime,
            }
        )

    def command(
        self, raw_input: str, name: str, args: list[str]
    ) -> ErrorMessage | None:
        """Record one successfully validated local command invocation.

        Args:
            raw_input: the exact prompt text submitted by the user.
            name: the parsed slash-command name or shell-command marker.
            args: the parsed command arguments.

        Returns:
            The temporary persistence error notice, or ``None`` on success.
        """
        return self.event(
            {
                "type": "command",
                "timestamp": self._utc_timestamp(),
                "raw_input": raw_input,
                "name": name,
                "args": args,
                "in_history": False,
                "visible": True,
            }
        )

    def close_interrupted_tool_calls(self) -> ErrorMessage | None:
        """Close model-history tool calls that have no durable paired result.

        A session that stopped mid-call leaves the model a request with no
        answer; each one gets a synthetic tool error so the model knows the
        outcome is unknown rather than seeing a dangling call.

        Returns:
            The first persistence error notice, or ``None`` when every synthetic
            result was saved or none was needed.
        """
        session = self._session()
        calls: dict[str, ToolCallMessage] = {}
        answered: set[str] = set()
        for event in session.events:
            record = event.record
            if record.get("in_history") is not True or not is_message_record(record):
                continue
            message = message_from_record(record)
            if isinstance(message, ToolCallMessage):
                calls[message.id] = message
            elif isinstance(message, ToolErrorMessage):
                answered.add(message.id)
            elif record.get("type") == "tool_result":
                answered.add(message.id)

        error: ErrorMessage | None = None
        for call_id in calls:
            if call_id in answered:
                continue
            error = error or self.message(
                ToolErrorMessage(
                    content=(
                        "No result was saved for this tool call before the session "
                        "stopped. The operation may or may not have completed."
                    ),
                    id=call_id,
                )
            )
        return error

    def event(self, record: dict[str, Any]) -> ErrorMessage | None:
        """Add one raw event to the open chat and try to persist it.

        Args:
            record: the JSON-compatible event to add.

        Returns:
            The temporary persistence error notice, or ``None`` on success.
        """
        self._session().add_event(record)
        return self.flush()

    def message_into(
        self, session: Session, message: Message, visible: bool = True
    ) -> ErrorMessage | None:
        """Record one finalized message into a session other than the open one.

        :meth:`message` always writes into whatever chat
        :attr:`SessionManager.current` reports; this is the one deliberate
        exception, used only when forking, where the same confirmation
        belongs in both the source session's log and the new fork's. It
        skips the file-state sync :meth:`message` does, since that concerns
        the *open* chat's tool reads, not a session that is no longer
        current.

        Args:
            session: the session to append the record to and flush.
            message: the message to persist.
            visible: whether the record belongs in the replayed UI log.

        Returns:
            The temporary persistence error notice, or ``None`` when the
            message was saved or is not meant to be.
        """
        record = message_to_record(message, visible)
        if record is None:
            return None
        session.add_event(record)
        return self.flush(session)

    def flush(self, session: Session | None = None) -> ErrorMessage | None:
        """Persist every unsaved event from a session in order.

        Every ordinary write reaches this method through :meth:`event`, for the
        open chat, so it is the one place the manager's registry is synced once
        a write lands. A fully successful flush also refreshes the log's
        checksum, certifying the clean state the next registry scan verifies.
        ``session`` is only ever passed explicitly by :meth:`message_into`, for
        a session that is not the open one.

        Args:
            session: the session to flush, or ``None`` for the open chat.

        Returns:
            The temporary persistence error notice, or ``None`` when all events are
            saved.
        """
        session = session or self._session()
        try:
            if session.recovery_offset is not None:
                self._storage_manager.truncate_file(
                    session.path, session.recovery_offset
                )
                session.recovery_offset = None

            for index, event in enumerate(session.events):
                if event.saved:
                    continue
                offset = self._storage_manager.file_size(session.path)
                line = json.dumps(
                    event.record, ensure_ascii=False, separators=(",", ":")
                )
                try:
                    if not session.created_on_disk and index == 0:
                        self._storage_manager.create_file(session.path, line)
                    else:
                        self._storage_manager.append_line(session.path, line)
                except OSError as exc:
                    if isinstance(exc, FileExistsError):
                        return self._save_error(exc)
                    self._handle_write_failure(session, index, offset)
                    return self._save_error(exc)
                event.saved = True
                session.created_on_disk = True
        except (OSError, TypeError, ValueError) as exc:
            return self._save_error(exc)

        if session.created_on_disk:
            SessionChecksum.write(session.path)
        self._session_manager.after_write(session)
        return None

    def _handle_write_failure(self, session: Session, index: int, offset: int) -> None:
        """Keep an event retryable or discard an incomplete first session file.

        Args:
            session: the session whose write failed.
            index: the position of the failed event in the session's events.
            offset: the file size observed before the failed write.
        """
        if not session.created_on_disk and index == 0:
            try:
                self._storage_manager.remove_file(session.path)
            except OSError:
                pass
            return
        session.recovery_offset = offset

    @staticmethod
    def _save_error(exc: Exception) -> ErrorMessage:
        """Return the notice a caller can render for a failed write.

        Args:
            exc: the filesystem or serialization failure.

        Returns:
            A renderable notice that is itself never persisted: the session is
            what failed.
        """
        return ErrorMessage(content=f"Could not save session: {exc}")

    @staticmethod
    def _utc_timestamp() -> str:
        """Return the current UTC timestamp in JSON-friendly ISO-8601 form.

        Returns:
            The current UTC time as an ISO-8601 string ending in ``Z``.
        """
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
