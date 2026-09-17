"""In-memory discovery index for persisted sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


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

    def __init__(self, entries: list[SessionRegistryEntry]) -> None:
        """Initialize the registry with the discovered entries.

        Args:
            entries: discovered session entries.
        """
        self.entries = list(entries)

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
