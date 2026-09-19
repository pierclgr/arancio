"""SHA-256 checksums letting the registry verify a session log without parsing it.

Every session log has a sibling checksum file recording the digest of the bytes that
were last fully flushed. The registry scan recomputes the digest and compares it with
the stored one instead of parsing the whole log: a match means the file is byte-
identical to a state that was cleanly validated, a mismatch means something changed it
afterwards.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from arancio.sessions.constants import SESSION_CHECKSUM_SUFFIX

_CHECKSUM_CHUNK_SIZE = 1 << 16


class SessionChecksum:
    """Read, verify and refresh the checksum file paired with a session log."""

    @staticmethod
    def path(log_path: Path) -> Path:
        """Return the checksum file paired with a session log.

        Args:
            log_path: the JSONL session log the checksum covers.

        Returns:
            The sibling checksum path next to the log.
        """
        return log_path.with_name(log_path.name + SESSION_CHECKSUM_SUFFIX)

    @classmethod
    def write(cls, log_path: Path) -> None:
        """Persist the checksum of a session log next to it.

        Args:
            log_path: the session log whose current bytes are checksumed.
        """
        cls.path(log_path).write_text(cls._digest(log_path), encoding="ascii")

    @classmethod
    def verify(cls, log_path: Path) -> bool | None:
        """Compare a session log's bytes with its recorded checksum.

        Args:
            log_path: the session log to verify.

        Returns:
            True when the file matches its recorded checksum, False on a
            mismatch, or ``None`` when no checksum file exists.
        """
        stored = cls.path(log_path)
        if not stored.exists():
            return None
        return stored.read_text(encoding="ascii").strip() == cls._digest(log_path)

    @staticmethod
    def _digest(path: Path) -> str:
        """Hash a file's bytes, reading it in chunks.

        Args:
            path: the file to hash.

        Returns:
            The lowercase hex SHA-256 digest of the file's content.
        """
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHECKSUM_CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()
