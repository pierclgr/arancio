"""Lifecycle, persistence and restoration for chat sessions."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from arancio.core.constants.path import ARANCIO_DEFAULT_DIR
from arancio.core.messages import (
    Message,
    UserMessage,
)
from arancio.core.tools.session import ToolSession, shared_session
from arancio.sessions.codec import (
    is_message_record,
    message_from_record,
)
from arancio.sessions.constants import ARANCIO_SESSIONS_DIR, SESSION_FORMAT_VERSION
from arancio.sessions.recorder import SessionRecorder
from arancio.sessions.registry import SessionRegistry, SessionRegistryEntry
from arancio.sessions.session import Session, SessionConfiguration
from arancio.storage.manager import StorageManager

if TYPE_CHECKING:
    from arancio.core.agents import Agent
    from arancio.settings.manager import SettingsManager


class SessionManager:
    """Create, persist, discover and restore the application's chat sessions."""

    def __init__(
        self,
        storage_manager: StorageManager,
        root: Path = ARANCIO_DEFAULT_DIR,
        tool_session: ToolSession | None = None,
    ) -> None:
        """Initialize the manager and discover existing sessions.

        Args:
            storage_manager: the file-only storage service for the arancio directory.
            root: the base directory containing the sessions directory.
            tool_session: the file-read safety state owned by the active chat.
        """
        self._storage_manager = storage_manager
        self._sessions_dir = root / ARANCIO_SESSIONS_DIR
        self._tool_session = tool_session or shared_session
        self._current: Session | None = None
        self._session_recorder = SessionRecorder(
            storage_manager, self, self._tool_session
        )
        self._registry = self.rebuild_registry()

    @property
    def current(self) -> Session | None:
        """Return the session active in this arancio process.

        Returns:
            The current session, or ``None`` before one is created or loaded.
        """
        return self._current

    @property
    def registry(self) -> SessionRegistry:
        """Return the in-memory index reconstructed from session files.

        Returns:
            Every discovered healthy and damaged session entry.
        """
        return self._registry

    @property
    def session_recorder(self) -> SessionRecorder:
        """Return the write side every caller records through.

        Returns:
            The recorder owning this manager's session log.
        """
        return self._session_recorder

    def after_write(self, session: Session) -> None:
        """Sync the registry once a write for the given session landed on disk.

        The recorder calls this from its flush, so the registry follows every
        durable write without any caller having to remember it. Both helpers are
        idempotent, so running them on every successful write is cheap.

        Args:
            session: the session whose write just succeeded.
        """
        self._register_durable_session(session)
        self._update_registry_entry(session)

    def rebuild_registry(self) -> SessionRegistry:
        """Discover every session file and mark damaged files without hiding them.

        Returns:
            The newly built registry containing every discovered session file.
        """
        entries = [
            self._entry_for_path(path)
            for path in self._storage_manager.find_files(
                self._storage_manager.make_dir(self._sessions_dir), "*.jsonl"
            )
        ]
        by_id: dict[str, list[SessionRegistryEntry]] = {}
        for entry in entries:
            by_id.setdefault(entry.id, []).append(entry)
        for duplicates in by_id.values():
            if len(duplicates) < 2:
                continue
            for entry in duplicates:
                other = next(item for item in duplicates if item is not entry)
                entry.status = "damaged"
                entry.damage_reason = f"Duplicate session ID found at {other.log_path}"
        self._registry = SessionRegistry(entries)
        return self._registry

    def create(
        self,
        working_directory: Path,
        configuration: SessionConfiguration,
    ) -> Session:
        """Create a new in-memory session and attempt to persist its first record.

        Args:
            working_directory: the absolute directory active for the new chat.
            configuration: the command-controlled configuration active at creation.

        Returns:
            The newly active session, even when its first disk save fails.
        """
        resolved_directory = working_directory.resolve()
        session_id = self._new_id()
        created_at = datetime.now().astimezone()
        session_path = (
            self._sessions_dir
            / self._directory_slug(resolved_directory)
            / created_at.strftime("%Y")
            / created_at.strftime("%m")
            / created_at.strftime("%d")
            / f"{session_id}.jsonl"
        )
        session = Session(
            id=session_id,
            name=session_id,
            created_at=created_at,
            creation_working_directory=resolved_directory,
            working_directory=resolved_directory,
            path=session_path,
            configuration=configuration,
            created_on_disk=False,
        )
        self._current = session
        self._session_recorder.created()
        return session

    def discard_and_create(
        self,
        working_directory: Path,
        configuration: SessionConfiguration,
    ) -> Session:
        """Discard the current in-memory session and begin a new chat session.

        Args:
            working_directory: the directory active for the new chat.
            configuration: the global configuration used by the new chat.

        Returns:
            The newly active session.
        """
        self._current = None
        self._tool_session.clear()
        return self.create(working_directory, configuration)

    def load(self, session_id: str) -> Session:
        """Load one healthy session from the reconstructed registry.

        Args:
            session_id: the globally unique session ID to restore.

        Returns:
            The newly active reconstructed session.

        Raises:
            ValueError: when the session is missing, ambiguous or damaged.
        """
        entry = self._registry.get(session_id)
        if entry.status == "damaged":
            raise ValueError(f"Session {session_id} is damaged: {entry.damage_reason}")
        self._current = self._read_session(entry.log_path)
        self._session_recorder.close_interrupted_tool_calls()
        return self._current

    def restore_runtime(
        self,
        session_id: str,
        agent: Agent,
        settings_manager: SettingsManager,
        launch_directory: Path,
    ) -> tuple[Path, str | None]:
        """Load a session and restore its non-UI runtime state.

        The caller applies the returned directory through the UI adapter so it
        can update both the process directory and the displayed toolbar.

        Args:
            session_id: the healthy session ID to restore.
            agent: the live agent whose model history is replaced.
            settings_manager: the live settings manager receiving the saved
                session configuration.
            launch_directory: the current launch directory used as a fallback
                when the saved directory disappeared.

        Returns:
            The usable working directory and an optional temporary fallback
            error for the TUI.
        """
        session = self.load(session_id)
        settings_manager.apply_session_configuration(session.configuration)
        agent.restore_history(self.model_history())
        self._tool_session.clear()
        self.restore_file_states()
        return self.restored_working_directory(launch_directory)

    def model_history(self) -> list[Message]:
        """Return the active session's messages that belong in model context.

        Returns:
            The normalized history messages in original event order.
        """
        return [
            message_from_record(event.record)
            for event in self.require_current().events
            if is_message_record(event.record)
            and event.record.get("in_history") is True
        ]

    def visible_messages(self) -> list[Message]:
        """Return the active session's messages that should be replayed in the TUI.

        A command has no message record of its own, so the line the user typed is
        rebuilt from the command record to hold its place in the log.

        Returns:
            The normalized visible messages in original event order.
        """
        messages: list[Message] = []
        for event in self.require_current().events:
            record = event.record
            if record.get("visible") is not True:
                continue
            if record.get("type") == "command":
                messages.append(UserMessage(content=record["raw_input"]))
            elif is_message_record(record):
                messages.append(message_from_record(record))
        return messages

    def restore_file_states(self) -> None:
        """Restore unchanged file-read safety records without reading file content."""
        for path, mtime in self.require_current().file_states.items():
            target = Path(path)
            if target.is_file() and target.stat().st_mtime == mtime:
                self._tool_session.record_read(path=path, mtime=mtime)

    def restored_working_directory(self, fallback: Path) -> tuple[Path, str | None]:
        """Return the saved directory or record a fallback when it no longer exists.

        Args:
            fallback: the process launch directory to use when the saved path
                no longer exists.

        Returns:
            The usable working directory and a temporary user-facing error when
            a fallback was necessary.
        """
        session = self.require_current()
        if session.working_directory.is_dir():
            return session.working_directory, None
        missing = session.working_directory
        resolved_fallback = fallback.resolve()
        save_error = self._session_recorder.working_directory(resolved_fallback)
        message = (
            f"Saved working directory no longer exists: {missing}; "
            f"using {resolved_fallback}."
        )
        if save_error:
            message = f"{message} {save_error.content}"
        return resolved_fallback, message

    def _entry_for_path(self, path: Path) -> SessionRegistryEntry:
        """Build one healthy or damaged registry entry from a session file.

        Args:
            path: the discovered JSONL session path.

        Returns:
            The healthy entry or a damaged entry with recovered metadata.
        """
        session_id = path.stem
        try:
            session = self._read_session(path)
        except (OSError, ValueError) as exc:
            name, working_directory = self._recover_entry_metadata(path, session_id)
            return SessionRegistryEntry(
                id=session_id,
                name=name,
                working_directory=working_directory,
                log_path=path,
                status="damaged",
                damage_reason=str(exc),
            )
        return SessionRegistryEntry(
            id=session.id,
            name=session.name,
            working_directory=session.working_directory,
            log_path=path,
            status="healthy",
        )

    def _register_durable_session(self, session: Session) -> None:
        """Add a newly durable session to the current in-memory registry.

        Args:
            session: the session whose creation record is now durable.
        """
        if not session.created_on_disk or any(
            entry.log_path == session.path for entry in self._registry.entries
        ):
            return
        self._registry.entries.append(
            SessionRegistryEntry(
                id=session.id,
                name=session.name,
                working_directory=session.working_directory,
                log_path=session.path,
                status="healthy",
            )
        )

    def _update_registry_entry(self, session: Session) -> None:
        """Synchronize the active durable session's mutable registry metadata.

        Args:
            session: the active session whose current CWD changed.
        """
        for entry in self._registry.entries:
            if entry.log_path == session.path:
                entry.working_directory = session.working_directory
                return

    def _read_session(self, path: Path) -> Session:
        """Parse and validate a complete JSONL session file.

        Args:
            path: the JSONL session file to parse.

        Returns:
            The fully reconstructed session.

        Raises:
            ValueError: when any line is malformed or violates the session schema.
        """
        lines = self._storage_manager.read_lines(path)
        if not lines:
            raise ValueError("Missing session_created record")

        records = [self._parse_line(line, index) for index, line in enumerate(lines, 1)]
        first = records[0]
        if first.get("type") != "session_created":
            raise ValueError("Missing session_created record")
        if first.get("format_version") != SESSION_FORMAT_VERSION:
            raise ValueError("Unsupported or missing session format version")

        session_id = self._required_string(first, "id")
        if session_id != path.stem:
            raise ValueError("Session ID does not match filename")
        name = self._required_string(first, "name")
        created_at = self._parse_datetime(self._required_string(first, "timestamp"))
        creation_directory = self._absolute_path(first, "creation_working_directory")
        working_directory = self._absolute_path(first, "working_directory")
        configuration = SessionConfiguration.from_dict(
            self._required_mapping(first, "configuration")
        )
        session = Session(
            id=session_id,
            name=name,
            created_at=created_at,
            creation_working_directory=creation_directory,
            working_directory=working_directory,
            path=path,
            configuration=configuration,
        )
        session.add_event(first, saved=True)
        for record in records[1:]:
            self._apply_record(session, record)
            session.add_event(record, saved=True)
        return session

    def _apply_record(self, session: Session, record: dict[str, Any]) -> None:
        """Validate one non-creation record and apply it to session state.

        Each record type states its own required fields and its own effect, so
        both live in the one branch that knows about that type.

        Args:
            session: the reconstructed session being updated.
            record: the JSON-compatible event record to validate and apply.

        Raises:
            ValueError: when the record type, its metadata or its state data is
                invalid.
        """
        self._parse_datetime(self._required_string(record, "timestamp"))
        record_type = record.get("type")

        if is_message_record(record):
            if not isinstance(record.get("in_history"), bool):
                raise ValueError("message in_history is missing or invalid")
            if not isinstance(record.get("visible"), bool):
                raise ValueError("message visible is missing or invalid")
            message_from_record(record)
            return
        if record_type == "command":
            self._required_string(record, "raw_input")
            self._required_string(record, "name")
            args = record.get("args")
            if not isinstance(args, list) or not all(
                isinstance(arg, str) for arg in args
            ):
                raise ValueError("command args are missing or invalid")
            if (
                record.get("in_history") is not False
                or record.get("visible") is not True
            ):
                raise ValueError("command visibility is invalid")
            return
        if record_type == "configuration_changed":
            session.configuration = SessionConfiguration.from_dict(
                self._required_mapping(record, "configuration")
            )
            return
        if record_type == "working_directory_changed":
            session.working_directory = self._absolute_path(record, "working_directory")
            return
        if record_type == "file_state_changed":
            file_path = self._absolute_path(record, "file_path")
            mtime = record.get("mtime")
            if not isinstance(mtime, (int, float)):
                raise ValueError("file state mtime is missing or invalid")
            session.file_states[str(file_path)] = float(mtime)
            return
        raise ValueError(f"Unknown session record type: {record_type!r}")

    def _recover_entry_metadata(
        self, path: Path, session_id: str
    ) -> tuple[str, Path | None]:
        """Recover safe registry metadata from a damaged session's first line.

        Args:
            path: the damaged JSONL session path.
            session_id: the ID derived from the filename.

        Returns:
            The recovered name and working directory, when available.
        """
        try:
            lines = self._storage_manager.read_lines(path)
            first = self._parse_line(lines[0], 1)
            name = first.get("name")
            working_directory = first.get("working_directory")
            return (
                name if isinstance(name, str) else session_id,
                Path(working_directory)
                if isinstance(working_directory, str)
                and Path(working_directory).is_absolute()
                else None,
            )
        except (IndexError, OSError, ValueError):
            return session_id, None

    def _new_id(self) -> str:
        """Generate an ID absent from the current in-memory registry.

        Returns:
            A new globally unique UUID4 hex string.
        """
        session_id = uuid.uuid4().hex
        while self._registry.contains(session_id):
            session_id = uuid.uuid4().hex
        return session_id

    @staticmethod
    def _directory_slug(path: Path) -> str:
        """Build a readable collision-resistant directory name for a CWD.

        Args:
            path: the resolved working directory.

        Returns:
            A filesystem-safe readable slug followed by a hash suffix.
        """
        raw_path = str(path)
        readable = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_path).strip("-")
        readable = (readable or "root")[:180]
        digest = hashlib.sha256(raw_path.encode("utf-8")).hexdigest()[:16]
        return f"--{readable}--{digest}"

    @staticmethod
    def _parse_line(line: str, line_number: int) -> dict[str, Any]:
        """Parse one JSONL object and retain its file line in errors.

        Args:
            line: the raw text line to parse.
            line_number: the one-indexed line number in the session file.

        Returns:
            The parsed JSON object.

        Raises:
            ValueError: when the line is invalid JSON or is not an object.
        """
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Session record at line {line_number} is not an object")
        return record

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        """Parse an aware ISO-8601 timestamp from a session record.

        Args:
            value: the serialized timestamp.

        Returns:
            The aware parsed timestamp.

        Raises:
            ValueError: when the timestamp is malformed or lacks a timezone.
        """
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("session timestamp is invalid") from exc
        if parsed.tzinfo is None:
            raise ValueError("session timestamp must include a timezone")
        return parsed

    @staticmethod
    def _required_string(record: dict[str, Any], field: str) -> str:
        """Return one required string record field.

        Args:
            record: the event record to inspect.
            field: the required field name.

        Returns:
            The field's string value.

        Raises:
            ValueError: when the field is absent or not a string.
        """
        value = record.get(field)
        if not isinstance(value, str):
            raise ValueError(f"{field} is missing or invalid")
        return value

    @classmethod
    def _absolute_path(cls, record: dict[str, Any], field: str) -> Path:
        """Return one required absolute path record field.

        Args:
            record: the event record to inspect.
            field: the required field name.

        Returns:
            The resolved absolute path.

        Raises:
            ValueError: when the field is absent, invalid or relative.
        """
        path = Path(cls._required_string(record, field))
        if not path.is_absolute():
            raise ValueError(f"{field} must be absolute")
        return path

    @staticmethod
    def _required_mapping(record: dict[str, Any], field: str) -> dict[str, Any]:
        """Return one required mapping record field.

        Args:
            record: the event record to inspect.
            field: the required field name.

        Returns:
            The field mapping.

        Raises:
            ValueError: when the field is absent or not a mapping.
        """
        value = record.get(field)
        if not isinstance(value, dict):
            raise ValueError(f"{field} is missing or invalid")
        return value

    def require_current(self) -> Session:
        """Return the active session or fail before an unscoped operation.

        Returns:
            The active session.

        Raises:
            ValueError: when no session is active in this process.
        """
        if self._current is None:
            raise ValueError("No active session.")
        return self._current
