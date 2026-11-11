"""Detached snapshot of one session's current state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from arancio.sessions.configurations import SessionConfiguration


@dataclass
class SessionState:
    """Detached current state without the managed event list or retry offset.

    Attributes:
        id: the conversation identity.
        created_at: the creation timestamp.
        creation_working_directory: the original working directory.
        working_directory: the current working directory.
        path: the fixed log location.
        configuration: an independent configuration snapshot.
        explicit_name: the chosen name, or None for the ID fallback.
        forked_from: the source identity, when forked.
        file_states: an independent saved path-to-mtime mapping.
        created_on_disk: whether the creation record was saved.
        has_pending_events: whether any records remain unsaved.
    """

    id: str
    created_at: datetime
    creation_working_directory: Path
    working_directory: Path
    path: Path
    configuration: SessionConfiguration
    explicit_name: str | None
    forked_from: str | None
    file_states: dict[str, float]
    created_on_disk: bool
    has_pending_events: bool

    @property
    def name(self) -> str:
        """The explicit name, or the ID when no name is set."""
        return self.id if self.explicit_name is None else self.explicit_name
