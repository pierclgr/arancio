"""Session discovery, full log validation and partial state-change validation.

Scans read the log to recover its last state and checksum status. Full loads also
validate every record. Both readers share the same state decoding rules.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.sessions.base import Session, SessionConfiguration, SessionScan
from arancio.sessions.checksum import SessionChecksum
from arancio.sessions.codec import is_message_record, message_from_record
from arancio.sessions.constants import SESSION_FORMAT_VERSION
from arancio.storage.manager import StorageManager


class SessionValidator:
    """Validate discovered logs, loaded records and requested state replacements.

    Attributes:
        __storage_manager: the supplied file I/O service.
        __categories: copied known permission-category identities.
        __checksum: the checksum helper using the same Storage service.
        __state_change_normalizers: editable names and validation/copy operations.
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        categories: Collection[PermissionCategory],
    ) -> None:
        """Store Storage and copy the category metadata used for decoding.

        Args:
            storage_manager: the generic file I/O service.
            categories: the known permission-category identities.
        """
        self.__storage_manager = storage_manager
        self.__categories = tuple(categories)
        self.__checksum = SessionChecksum(storage_manager)
        self.__state_change_normalizers: dict[str, Callable[[Any], Any]] = {
            "configuration": self.__normalize_configuration,
            "working_directory": self.__normalize_directory,
            "explicit_name": self.__normalize_name,
        }

    def scan(self, path: Path) -> SessionScan:
        """Inspect one session log without parsing it fully.

        Statuses: ``healthy`` when the checksum matches the file, ``unverified``
        when no checksum exists, ``damaged`` when scanning or checksum
        verification fails.

        Args:
            path: the discovered JSONL session path.

        Returns:
            The scan result with recovered metadata and integrity status.
        """
        configuration = None
        working_directory = None
        name = None
        try:
            configuration, working_directory, name = self.__scan_state(path)
            matches = self.__checksum.verify(path)
        except (OSError, ValueError) as exc:
            return SessionScan(
                explicit_name=name,
                working_directory=working_directory,
                configuration=configuration,
                status="damaged",
                damage_reason=str(exc),
            )
        if matches is None:
            return SessionScan(
                explicit_name=name,
                working_directory=working_directory,
                configuration=configuration,
                status="unverified",
            )
        if matches:
            return SessionScan(
                explicit_name=name,
                working_directory=working_directory,
                configuration=configuration,
                status="healthy",
            )
        return SessionScan(
            explicit_name=name,
            working_directory=working_directory,
            configuration=configuration,
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
        lines = self.__storage_manager.read_lines(path)
        if not lines:
            raise ValueError("Missing session_created record")

        records = [
            self.__parse_line(line, index) for index, line in enumerate(lines, 1)
        ]
        first = records[0]
        if first.get("type") != "session_created":
            raise ValueError("Missing session_created record")
        if first.get("format_version") != SESSION_FORMAT_VERSION:
            raise ValueError("Unsupported or missing session format version")

        session_id = self.__required_string(first, "id")
        if session_id != path.stem:
            raise ValueError("Session ID does not match filename")
        created_at = self.__parse_datetime(self.__required_string(first, "timestamp"))
        creation_directory = self.__absolute_path(first, "creation_working_directory")
        # not part of a state_changed snapshot, so read directly off the header
        forked_from = first.get("forked_from")
        if "forked_from" not in first or (
            forked_from is not None and not isinstance(forked_from, str)
        ):
            raise ValueError("forked_from is missing or invalid")

        self.__state_from_record(first)
        latest_state_record = first
        for record in records[1:]:
            if record.get("type") == "state_changed":
                latest_state_record = record
        configuration, working_directory, name = self.__state_from_record(
            latest_state_record
        )

        session = Session(
            id=session_id,
            explicit_name=name,
            forked_from=forked_from,
            created_at=created_at,
            creation_working_directory=creation_directory,
            working_directory=working_directory,
            path=path,
            configuration=configuration,
        )
        session.add_event(first, saved=True)
        for record in records[1:]:
            self.__apply_record(session, record)
            session.add_event(record, saved=True)
        return session

    def validate_state_changes(self, changes: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and copy supplied replacements without changing a Session.

        Args:
            changes: supplied editable fields; omitted fields are unchanged.

        Returns:
            Validated replacements with independent mutable values.

        Raises:
            ValueError: when a field name or its value is invalid.
        """
        replacements: dict[str, Any] = {}
        for field, value in changes.items():
            if field not in self.__state_change_normalizers:
                raise ValueError(f"Unknown editable session field {field!r}: {value!r}")
            normalize = self.__state_change_normalizers[field]
            replacements[field] = normalize(value)
        return replacements

    def __normalize_configuration(self, value: Any) -> SessionConfiguration:
        """Validate and copy a typed session configuration.

        Args:
            value: the proposed configuration replacement.

        Returns:
            A checked configuration with independent permission values.

        Raises:
            ValueError: when the replacement is not a configuration object.
        """
        if not isinstance(value, SessionConfiguration):
            raise ValueError(f"Invalid session configuration: {value!r}")
        if not isinstance(value.permissions, Mapping):
            raise ValueError(
                f"Invalid configuration permissions: {value.permissions!r}"
            )
        for category, level in value.permissions.items():
            if category not in self.__categories:
                raise ValueError(
                    f"Unknown configuration permission category: {category!r}"
                )
            if not isinstance(level, PermissionLevel):
                raise ValueError(f"Invalid configuration permission level: {level!r}")
        return SessionConfiguration.from_dict(value.to_dict(), self.__categories)

    @staticmethod
    def __normalize_directory(value: Any) -> Path:
        """Require an absolute working-directory value without filesystem checks.

        Args:
            value: the proposed directory replacement.

        Returns:
            The supplied absolute Path.

        Raises:
            ValueError: when the value is not an absolute Path.
        """
        if not isinstance(value, Path) or not value.is_absolute():
            raise ValueError(f"Expected an absolute working_directory Path: {value!r}")
        return value

    @staticmethod
    def __normalize_name(value: Any) -> str | None:
        """Require a string name or an explicit None to clear it.

        Args:
            value: the proposed explicit name.

        Returns:
            The supplied string or None.

        Raises:
            ValueError: when the name is neither a string nor None.
        """
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Invalid session explicit_name: {value!r}")
        return value

    def certify(self, path: Path) -> None:
        """Persist the checksum of a session log that just validated cleanly.

        Args:
            path: the session log whose current bytes are checksumed.
        """
        self.__checksum.write(path)

    def __scan_state(self, path: Path) -> tuple[SessionConfiguration, Path, str | None]:
        """Recover a session's latest state snapshot without a full parse.

        Reads every line, but only inspects each one's ``type`` to find the
        winning record; only that one record's configuration, working
        directory and name are actually validated and built, unlike
        :meth:`read`, which validates every line's own schema too.

        Args:
            path: the JSONL session path.

        Returns:
            The configuration, working directory and name of the last
            ``state_changed`` record, or the header's own when none exists.

        Raises:
            ValueError: when the first line is missing, not a session header,
                or the winning state record is invalid.
        """
        lines = self.__storage_manager.read_lines(path)
        if not lines:
            raise ValueError("Missing session_created record")
        first = self.__parse_line(lines[0], 1)
        if first.get("type") != "session_created":
            raise ValueError("Missing session_created record")

        latest = first
        for index, line in enumerate(lines[1:], 2):
            record = self.__parse_line(line, index)
            if record.get("type") == "state_changed":
                latest = record
        return self.__state_from_record(latest)

    def __state_from_record(
        self,
        record: dict[str, Any],
    ) -> tuple[SessionConfiguration, Path, str | None]:
        """Build the configuration, working directory and name a state snapshot carries.

        Shared by :meth:`__scan_state` and :meth:`read`, both of which need to
        turn a ``session_created`` or ``state_changed`` record into the same
        three values.

        Args:
            record: the winning ``session_created`` or ``state_changed`` record.

        Returns:
            The record's configuration, absolute working directory and explicit name.

        Raises:
            ValueError: when the name is absent or neither a string nor null.
        """
        configuration = SessionConfiguration.from_dict(
            self.__required_mapping(record, "configuration"),
            self.__categories,
        )
        working_directory = self.__absolute_path(record, "working_directory")
        name = record.get("name")
        if "name" not in record:
            raise ValueError("name is missing")
        return configuration, working_directory, self.__normalize_name(name)

    def __apply_record(self, session: Session, record: dict[str, Any]) -> None:
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
        self.__parse_datetime(self.__required_string(record, "timestamp"))
        record_type = record.get("type")

        if is_message_record(record):
            if not isinstance(record.get("in_history"), bool):
                raise ValueError("message in_history is missing or invalid")
            if not isinstance(record.get("visible"), bool):
                raise ValueError("message visible is missing or invalid")
            message_from_record(record)
            return
        if record_type == "command":
            self.__required_string(record, "raw_input")
            self.__required_string(record, "name")
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
            # validated here so every line is checked; the final value came
            # from the winning record already, resolved once in `read`
            self.__state_from_record(record)
            return
        if record_type == "file_state_changed":
            file_path = self.__absolute_path(record, "file_path")
            mtime = record.get("mtime")
            if not isinstance(mtime, (int, float)):
                raise ValueError("file state mtime is missing or invalid")
            session.file_states[str(file_path)] = float(mtime)
            return
        raise ValueError(f"Unknown session record type: {record_type!r}")

    @staticmethod
    def __parse_line(line: str, line_number: int) -> dict[str, Any]:
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
    def __parse_datetime(value: str) -> datetime:
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
    def __required_string(record: dict[str, Any], field: str) -> str:
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
    def __absolute_path(cls, record: dict[str, Any], field: str) -> Path:
        """Return one required absolute path record field.

        Args:
            record: the event record to inspect.
            field: the required field name.

        Returns:
            The resolved absolute path.

        Raises:
            ValueError: when the field is absent, invalid or relative.
        """
        path = Path(cls.__required_string(record, field))
        if not path.is_absolute():
            raise ValueError(f"{field} must be absolute")
        return path

    @staticmethod
    def __required_mapping(record: dict[str, Any], field: str) -> dict[str, Any]:
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
