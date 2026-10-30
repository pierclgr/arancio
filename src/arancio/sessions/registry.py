"""In-memory discovery index rebuilt from saved session logs.

Scanning reads every JSONL record to locate the final state, validates that state, and
checks the log checksum. Full record validation is deferred until loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from arancio.sessions.base import Session, SessionRegistryEntry

if TYPE_CHECKING:
    from arancio.sessions.validator import SessionValidator
    from arancio.storage.manager import StorageManager


class SessionRegistry:
    """Index saved sessions without persisting a separate registry.

    Attributes:
        __storage_manager: the service discovering session logs.
        __validator: the scanner recovering metadata and integrity status.
        __entries: the owned entries in deterministic discovery order.
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        sessions_dir: Path,
        validator: SessionValidator,
    ) -> None:
        """Discover the supplied session directory.

        Args:
            storage_manager: the generic file I/O service.
            sessions_dir: the explicit directory containing session logs.
            validator: the scanner used for each discovered file.
        """
        self.__storage_manager = storage_manager
        self.__validator = validator
        self.__entries: list[SessionRegistryEntry] = []
        self.build(sessions_dir)

    def build(self, sessions_dir: Path) -> None:
        """Replace the index with a deterministic scan, marking duplicate IDs.

        Args:
            sessions_dir: the directory tree containing session logs.
        """
        self.__storage_manager.create_directory(sessions_dir)
        entries: list[SessionRegistryEntry] = []
        first_by_id: dict[str, SessionRegistryEntry] = {}
        duplicate_ids: set[str] = set()
        # scan each log and compare its id with the first entry seen
        for path in self.__storage_manager.find_files(sessions_dir, "*.jsonl"):
            entry = self.__create_entry(path)
            entries.append(entry)
            first = first_by_id.get(entry.id)
            if first is None:
                first_by_id[entry.id] = entry
                continue
            # keep the first entry's reason pointing to the first duplicate
            if entry.id not in duplicate_ids:
                first.status = "damaged"
                first.damage_reason = f"Duplicate session ID found at {entry.log_path}"
                duplicate_ids.add(entry.id)
            entry.status = "damaged"
            entry.damage_reason = f"Duplicate session ID found at {first.log_path}"
        self.__entries = entries

    def contains(self, session_id: str) -> bool:
        """Return whether the index contains an identity.

        Args:
            session_id: the identity to look up.

        Returns:
            True when at least one discovered entry has this identity.
        """
        return any(entry.id == session_id for entry in self.__entries)

    def get(self, session_id: str) -> SessionRegistryEntry:
        """Return one unambiguous entry for internal loading.

        Args:
            session_id: the requested saved-session identity.

        Returns:
            The matching internal registry entry.

        Raises:
            ValueError: when no entry exists or the identity is duplicated.
        """
        matches = [entry for entry in self.__entries if entry.id == session_id]
        if not matches:
            raise ValueError(f"Session not found: {session_id}")
        if len(matches) > 1:
            raise ValueError(f"Session ID is ambiguous: {session_id}")
        return matches[0]

    def find(self, query: str) -> list[SessionRegistryEntry]:
        """Find exact identity matches, otherwise case-insensitive name matches.

        Duplicate exact IDs remain separate results. The manager copies entries
        before returning them outside Sessions.

        Args:
            query: an identity or name substring.

        Returns:
            All matches in registry order, including damaged entries.
        """
        exact = [entry for entry in self.__entries if entry.id == query]
        if exact:
            return exact
        needle = query.lower()
        return [entry for entry in self.__entries if needle in entry.name.lower()]

    def update(self, session: Session) -> None:
        """Publish independent metadata after a durable write or valid load.

        Args:
            session: the managed session whose metadata is now durable.
        """
        if not session.created_on_disk:
            return
        for entry in self.__entries:
            if entry.log_path == session.path:
                entry.working_directory = session.working_directory
                entry.configuration = session.configuration.copy()
                entry.explicit_name = session.explicit_name
                entry.status = "healthy"
                entry.damage_reason = None
                return
        self.__entries.append(
            SessionRegistryEntry(
                id=session.id,
                working_directory=session.working_directory,
                configuration=session.configuration.copy(),
                log_path=session.path,
                status="healthy",
                explicit_name=session.explicit_name,
            )
        )

    def __create_entry(self, path: Path) -> SessionRegistryEntry:
        """Build an entry from the filename and validator's scan result.

        Args:
            path: the discovered JSONL path.

        Returns:
            The recovered metadata and discovery status.
        """
        scan = self.__validator.scan(path)
        return SessionRegistryEntry(
            id=path.stem,
            explicit_name=scan.explicit_name,
            working_directory=scan.working_directory,
            configuration=scan.configuration,
            log_path=path,
            status=scan.status,
            damage_reason=scan.damage_reason,
        )
