"""Build and persist session records through Storage, without live Core state."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from arancio.core.messages.base import ErrorMessage, Message
from arancio.core.messages.tool import (
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
)
from arancio.sessions.base import Session
from arancio.sessions.checksum import SessionChecksum
from arancio.sessions.codec import (
    is_message_record,
    message_from_record,
    message_to_record,
)
from arancio.sessions.constants import SESSION_FORMAT_VERSION
from arancio.storage.manager import StorageManager


class SessionRecorder:
    """Own record creation, pending writes, recovery and checksum refresh.

    Attributes:
        __storage_manager: the file I/O service writing logs.
        __get_current: the manager's private active-session callback.
        __after_write: the manager's private successful-write callback.
        __checksum: the helper hashing logs through the same Storage service.
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        get_current: Callable[[], Session],
        after_write: Callable[[Session], None],
    ) -> None:
        """Store dependencies without reading a partly constructed manager.

        Args:
            storage_manager: the generic file I/O service.
            get_current: the callback returning the managed active session.
            after_write: the callback publishing durable session metadata.
        """
        self.__storage_manager = storage_manager
        self.__get_current = get_current
        self.__after_write = after_write
        self.__checksum = SessionChecksum(storage_manager)

    def created(self) -> None:
        """Queue the opening header without creating a conversation log."""
        session = self.__get_current()
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

    def message(
        self,
        message: Message,
        *,
        file_states: Mapping[str, float],
        visible: bool = True,
    ) -> ErrorMessage | None:
        """Queue one supported message and changed read states, then flush.

        Chunks are skipped. Unsupported message types raise ValueError through
        the codec. Message and read-state records remain queued on save failure.

        Args:
            message: the normalized message to save.
            file_states: the supplied absolute-path-to-mtime snapshot.
            visible: whether this message belongs in visible replay.

        Returns:
            A temporary save notice, or None when saved or skipped.
        """
        record = message_to_record(message, visible)
        if record is None:
            return None
        self.__get_current().add_event(record)
        self.__sync_file_states(file_states)
        return self.flush()

    def __sync_file_states(self, snapshot: Mapping[str, float]) -> None:
        """Queue changed supplied reads without accessing a live Core guard.

        Args:
            snapshot: the independent path-to-mtime mapping supplied by the caller.
        """
        session = self.__get_current()
        for path, mtime in snapshot.items():
            if session.file_states.get(path) == mtime:
                continue
            session.add_event(
                {
                    "type": "file_state_changed",
                    "timestamp": self.__utc_timestamp(),
                    "file_path": path,
                    "mtime": mtime,
                }
            )
            session.file_states[path] = mtime

    def queue_state(self) -> None:
        """Queue the complete active state without flushing it.

        Forking uses this after copied history so old source state records cannot
        replace the child's name on reload. An unused fork still writes nothing.
        """
        session = self.__get_current()
        session.add_event(
            {
                "type": "state_changed",
                "timestamp": self.__utc_timestamp(),
                "configuration": session.configuration.to_dict(),
                "working_directory": str(session.working_directory),
                "name": session.explicit_name,
            }
        )

    def state_changed(self) -> ErrorMessage | None:
        """Queue the complete current state and attempt persistence.

        The manager has already changed the managed fields. A failed save leaves
        those fields and their queued record available for retry.

        Returns:
            A temporary save notice, or None on success.
        """
        self.queue_state()
        return self.flush()

    def command(
        self, raw_input: str, name: str, args: list[str]
    ) -> ErrorMessage | None:
        """Record the original typed command and copied parsed arguments.

        Args:
            raw_input: the exact input supplied by the user.
            name: the parsed command name or shell marker.
            args: the parsed argument list.

        Returns:
            A temporary save notice, or None on success.
        """
        return self.__event(
            {
                "type": "command",
                "timestamp": self.__utc_timestamp(),
                "raw_input": raw_input,
                "name": name,
                "args": args,
                "in_history": False,
                "visible": True,
            }
        )

    def close_interrupted_tool_calls(self) -> ErrorMessage | None:
        """Queue all missing tool outcomes before attempting their repair write.

        No tool is executed. Every unmatched model-history call receives the
        existing synthetic error explaining that its previous outcome is unknown.

        Returns:
            A temporary repair-save notice, or None when saved or unnecessary.
        """
        session = self.__get_current()
        calls: dict[str, ToolCallMessage] = {}
        answered: set[str] = set()
        for event in session.events:
            record = event.record
            if record.get("in_history") is not True or not is_message_record(record):
                continue
            message = message_from_record(record)
            if isinstance(message, ToolCallMessage):
                calls[message.id] = message
            elif isinstance(message, ToolResultMessage):
                answered.add(message.id)

        repaired = False
        for call_id in calls:
            if call_id in answered:
                continue
            error = ToolErrorMessage(
                content=(
                    "No result was saved for this tool call before the session "
                    "stopped. The operation may or may not have completed."
                ),
                id=call_id,
            )
            session.add_event(message_to_record(error))
            repaired = True
        if repaired:
            return self.flush()
        return None

    def __event(self, record: dict[str, Any]) -> ErrorMessage | None:
        """Queue one owned record and attempt persistence.

        Args:
            record: the JSON-compatible record to append.

        Returns:
            A temporary save notice, or None on success.
        """
        self.__get_current().add_event(record)
        return self.flush()

    def message_into(
        self, session: Session, message: Message, visible: bool = True
    ) -> ErrorMessage | None:
        """Record a fork confirmation without changing the active session.

        This deliberate write does not synchronize file-read state. Checksum
        failures propagate after the log write, as in an ordinary flush.

        Args:
            session: the source or child receiving the confirmation.
            message: the finalized message to save.
            visible: whether it belongs in visible replay.

        Returns:
            A temporary save notice, or None when saved or skipped.
        """
        record = message_to_record(message, visible)
        if record is None:
            return None
        session.add_event(record)
        return self.flush(session)

    def flush(self, session: Session | None = None) -> ErrorMessage | None:
        """Persist queued records in order and certify a clean log.

        Header-only new conversations stay in memory. Log failures return a save
        notice; checksum failures propagate. The manager is notified only after
        the checksum write succeeds.

        Args:
            session: an explicit fork source, or None for the active session.

        Returns:
            A temporary save notice, or None when complete or unused.
        """
        session = session if session is not None else self.__get_current()
        if not session.created_on_disk and len(session.events) == 1:
            return None
        try:
            if session.recovery_offset is not None:
                self.__storage_manager.truncate_file(
                    session.path, session.recovery_offset
                )
                session.recovery_offset = None

            for index, event in enumerate(session.events):
                if event.saved:
                    continue
                first_write = not session.created_on_disk and index == 0
                offset = (
                    0
                    if first_write
                    else self.__storage_manager.get_file_size(session.path)
                )
                line = json.dumps(
                    event.record, ensure_ascii=False, separators=(",", ":")
                )
                try:
                    if first_write:
                        self.__storage_manager.create_file(session.path, line)
                    else:
                        self.__storage_manager.append_line(session.path, line)
                except OSError as exc:
                    if not isinstance(exc, FileExistsError):
                        self.__handle_write_failure(session, index, offset)
                    return self.__save_error(exc)
                event.saved = True
                session.created_on_disk = True
        except (OSError, TypeError, ValueError) as exc:
            return self.__save_error(exc)

        if session.created_on_disk:
            self.__checksum.write(session.path)
            self.__after_write(session)
        return None

    def __handle_write_failure(self, session: Session, index: int, offset: int) -> None:
        """Retain append recovery state or remove an incomplete new file.

        Args:
            session: the session whose write failed.
            index: the failed event's position.
            offset: the byte count before the write.
        """
        if not session.created_on_disk and index == 0:
            try:
                self.__storage_manager.remove_file(session.path)
            except OSError:
                pass
            return
        session.recovery_offset = offset

    @staticmethod
    def __save_error(exc: Exception) -> ErrorMessage:
        """Build a notice that is never passed back through the failed recorder.

        Args:
            exc: the storage or serialization failure.

        Returns:
            The temporary user-facing save notice.
        """
        return ErrorMessage(content=f"Could not save session: {exc}")

    @staticmethod
    def __utc_timestamp() -> str:
        """Return the current UTC timestamp for a record.

        Returns:
            An aware ISO timestamp ending in Z.
        """
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
