"""In-memory model of one persistent chat session."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from arancio.sessions.configurations import SessionConfiguration
from arancio.sessions.events.base import SessionEvent


@dataclass
class Session:
    """One managed chat session and the events needed to restore it.

    A session forked from another carries the source session's ID in ``forked_from``; it
    is ``None`` for a session created or resumed normally.
    """

    id: str
    created_at: datetime
    creation_working_directory: Path
    working_directory: Path
    path: Path
    configuration: SessionConfiguration
    explicit_name: str | None = None
    forked_from: str | None = None
    events: list[SessionEvent] = field(default_factory=list)
    recovery_offset: int | None = None
    created_on_disk: bool = True
    file_states: dict[str, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """The explicit name, or the ID when no name is set."""
        return self.id if self.explicit_name is None else self.explicit_name

    @name.setter
    def name(self, value: str | None) -> None:
        """Set the explicit name, or clear it with ``None``.

        Args:
            value: the explicit name to store.
        """
        self.explicit_name = value

    def add_event(self, record: dict[str, Any], saved: bool = False) -> SessionEvent:
        """Append one event to this session's ordered log.

        Args:
            record: the JSON-compatible event record.
            saved: whether the event is already durable on disk.

        Returns:
            The newly appended in-memory event.
        """
        event = SessionEvent(record=deepcopy(record), saved=saved)
        self.events.append(event)
        return event
