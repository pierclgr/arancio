"""In-memory discovery index for persisted sessions.

The scan stays cheap: the validator inspects only a log's header and its checksum, so no
session is fully parsed until it is actually loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from arancio.sessions.session import SessionConfiguration
    from arancio.sessions.validator import SessionValidator
    from arancio.storage.manager import StorageManager


@dataclass
class SessionRegistryEntry:
    """One discovered session file and its restoration status."""

    id: str
    name: str
    working_directory: Path | None
    configuration: SessionConfiguration | None
    log_path: Path
    status: Literal["healthy", "unverified", "damaged"]
    damage_reason: str | None = None


class SessionRegistry:
    """Index discovered session files without persisting a second registry file."""

    def __init__(
        self,
        storage_manager: StorageManager,
        sessions_dir: Path,
        validator: SessionValidator,
    ) -> None:
        """Initialize the registry by discovering every session file.

        Args:
            storage_manager: the file-only storage service scanning the log files.
            sessions_dir: the directory tree containing the session logs.
            validator: checks each discovered log's header and checksum.
        """
        self._storage_manager = storage_manager
        self._validator = validator
        self.entries: list[SessionRegistryEntry] = []
        self.build(sessions_dir)

    def build(self, sessions_dir: Path) -> SessionRegistry:
        """Discover every session file and populate the index.

        Damaged files are listed rather than hidden, and an ID that maps to more
        than one file damages both copies because neither can be trusted to be
        the real chat.

        Args:
            sessions_dir: the directory tree containing the session logs.

        Returns:
            This registry, holding every discovered entry.
        """
        entries = [
            self._entry_for_path(path)
            for path in self._storage_manager.find_files(
                self._storage_manager.make_dir(sessions_dir), "*.jsonl"
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
        self.entries = entries
        return self

    def contains(self, session_id: str) -> bool:
        """Return whether any discovered session uses an ID.

        Args:
            session_id: the ID to look up.

        Returns:
            True when at least one entry has the requested ID.
        """
        return any(entry.id == session_id for entry in self.entries)

    def get(self, session_id: str) -> SessionRegistryEntry:
        """Return one unambiguous session entry by ID.

        Args:
            session_id: the requested session ID.

        Returns:
            The matching registry entry.

        Raises:
            ValueError: when the ID is absent or maps to more than one file.
        """
        matches = [entry for entry in self.entries if entry.id == session_id]
        if not matches:
            raise ValueError(f"Session not found: {session_id}")
        if len(matches) > 1:
            raise ValueError(f"Session ID is ambiguous: {session_id}")
        return matches[0]

    def _entry_for_path(self, path: Path) -> SessionRegistryEntry:
        """Build one registry entry from a session file's scan result.

        Args:
            path: the discovered JSONL session path.

        Returns:
            The entry holding the log's metadata and checksum-based status.
        """
        scan = self._validator.scan(path)
        return SessionRegistryEntry(
            id=path.stem,
            name=scan.name,
            working_directory=scan.working_directory,
            configuration=scan.configuration,
            log_path=path,
            status=scan.status,
            damage_reason=scan.damage_reason,
        )
