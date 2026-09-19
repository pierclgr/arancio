"""In-memory discovery index for persisted sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from arancio.sessions.session import Session
    from arancio.storage.manager import StorageManager


@dataclass
class SessionRegistryEntry:
    """One discovered session file and its restoration status."""

    id: str
    name: str
    working_directory: Path | None
    log_path: Path
    status: Literal["healthy", "damaged"]
    damage_reason: str | None = None


class SessionRegistry:
    """Index discovered session files without persisting a second registry file."""

    def __init__(
        self,
        storage_manager: StorageManager,
        sessions_dir: Path,
        read_session: Callable[[Path], Session],
    ) -> None:
        """Initialize the registry by discovering every session file.

        Args:
            storage_manager: the file-only storage service scanning the log files.
            sessions_dir: the directory tree containing the session logs.
            read_session: parses and validates one complete session log.
        """
        self._storage_manager = storage_manager
        self.entries: list[SessionRegistryEntry] = []
        self.build(sessions_dir, read_session)

    def build(
        self, sessions_dir: Path, read_session: Callable[[Path], Session]
    ) -> SessionRegistry:
        """Discover every session file and populate the index.

        Damaged files are listed rather than hidden, and an ID that maps to more
        than one file damages both copies because neither can be trusted to be
        the real chat.

        Args:
            sessions_dir: the directory tree containing the session logs.
            read_session: parses and validates one complete session log.

        Returns:
            This registry, holding every discovered entry.
        """
        entries = [
            self._entry_for_path(path, read_session)
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

    @classmethod
    def _entry_for_path(
        cls, path: Path, read_session: Callable[[Path], Session]
    ) -> SessionRegistryEntry:
        """Build one healthy or damaged registry entry from a session file.

        Args:
            path: the discovered JSONL session path.
            read_session: parses and validates one complete session log.

        Returns:
            The healthy entry or a damaged entry with recovered metadata.
        """
        session_id = path.stem
        try:
            session = read_session(path)
        except (OSError, ValueError) as exc:
            name, working_directory = cls._recover_entry_metadata(path, session_id)
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

    @staticmethod
    def _recover_entry_metadata(path: Path, session_id: str) -> tuple[str, Path | None]:
        """Recover safe registry metadata from a damaged session's first line.

        Args:
            path: the damaged JSONL session path.
            session_id: the ID derived from the filename.

        Returns:
            The recovered name and working directory, when available.
        """
        try:
            first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            if not isinstance(first, dict):
                return session_id, None
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
