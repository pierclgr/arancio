"""Validation of session logs: cheap checksum scan, full validation at load.

A session is inspected at two moments with different depth. The registry scan wants only
metadata and a cheap integrity signal, so it parses the header line and compares the
checksum written after the last clean flush. Loading wants to trust the content itself,
so it parses and validates every line. Both live in one class so the two layers can
never drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from arancio.sessions.checksum import SessionChecksum
from arancio.sessions.codec import is_message_record, message_from_record
from arancio.sessions.constants import SESSION_FORMAT_VERSION
from arancio.sessions.session import Session, SessionConfiguration
from arancio.storage.manager import StorageManager


@dataclass
class SessionScan:
    """What the registry needs from one session log: metadata plus integrity."""

    name: str
    working_directory: Path | None
    status: Literal["healthy", "unverified", "damaged"]
    damage_reason: str | None = None


class SessionValidator:
    """Check every discovered session cheaply, and one loaded session fully."""

    def __init__(self, storage_manager: StorageManager) -> None:
        """Initialize the validator with the storage service it reads through.

        Args:
            storage_manager: the file-only storage service reading session logs.
        """
        self._storage_manager = storage_manager

    def scan(self, path: Path) -> SessionScan:
        """Inspect one session log without parsing it fully.

        Statuses: ``healthy`` when the checksum matches the file, ``unverified``
        when no checksum exists, ``damaged`` when the checksum mismatches or
        the header cannot be trusted as a session header.

        Args:
            path: the discovered JSONL session path.

        Returns:
            The scan result with recovered metadata and integrity status.
        """
        session_id = path.stem
        try:
            name, working_directory = self._read_header(path, session_id)
        except (OSError, ValueError) as exc:
            return SessionScan(
                name=session_id,
                working_directory=None,
                status="damaged",
                damage_reason=str(exc),
            )
        matches = SessionChecksum.verify(path)
        if matches is None:
            return SessionScan(
                name=name, working_directory=working_directory, status="unverified"
            )
        if matches:
            return SessionScan(
                name=name, working_directory=working_directory, status="healthy"
            )
        return SessionScan(
            name=name,
            working_directory=working_directory,
            status="damaged",
            damage_reason="Session log no longer matches its checksum",
        )

    def read(self, path: Path) -> Session:
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

    def certify(self, path: Path) -> None:
        """Persist the checksum of a session log that just validated cleanly.

        Args:
            path: the session log whose current bytes are checksumed.
        """
        SessionChecksum.write(path)

    @staticmethod
    def _read_header(path: Path, session_id: str) -> tuple[str, Path | None]:
        """Parse a session's display metadata from its first line only.

        Args:
            path: the JSONL session path.
            session_id: the ID derived from the filename, used as a fallback name.

        Returns:
            The header's name and working directory, when available.

        Raises:
            ValueError: when the first line is missing or not a session header.
        """
        with path.open("r", encoding="utf-8") as handle:
            line = handle.readline()
        if not line:
            raise ValueError("Missing session_created record")
        try:
            first = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON at line 1") from exc
        if not isinstance(first, dict):
            raise ValueError("Session record at line 1 is not an object")
        if first.get("type") != "session_created":
            raise ValueError("Missing session_created record")
        name = first.get("name")
        working_directory = first.get("working_directory")
        return (
            name if isinstance(name, str) else session_id,
            Path(working_directory)
            if isinstance(working_directory, str)
            and Path(working_directory).is_absolute()
            else None,
        )

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
        if record_type == "state_changed":
            session.configuration = SessionConfiguration.from_dict(
                self._required_mapping(record, "configuration")
            )
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
