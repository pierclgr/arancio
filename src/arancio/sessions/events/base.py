"""In-memory events of one session log."""

from dataclasses import dataclass
from typing import Any


@dataclass
class SessionEvent:
    """One JSONL event and its in-process persistence state."""

    record: dict[str, Any]
    saved: bool = True
