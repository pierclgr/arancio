"""Registry entry for one saved session file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from arancio.sessions.configurations import SessionConfiguration


@dataclass
class SessionRegistryEntry:
    """Search metadata and discovery status for one saved session file."""

    id: str
    working_directory: Path | None
    configuration: SessionConfiguration | None
    log_path: Path
    status: Literal["healthy", "unverified", "damaged"]
    explicit_name: str | None = None
    damage_reason: str | None = None

    @property
    def name(self) -> str:
        """The explicit name, or the ID when no name is set."""
        return self.id if self.explicit_name is None else self.explicit_name
