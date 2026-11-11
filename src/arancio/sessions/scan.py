"""Validator result for one scanned session file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from arancio.sessions.configurations import SessionConfiguration


@dataclass
class SessionScan:
    """Temporary validator result used to build a registry entry."""

    explicit_name: str | None
    working_directory: Path | None
    configuration: SessionConfiguration | None
    status: Literal["healthy", "unverified", "damaged"]
    damage_reason: str | None = None
