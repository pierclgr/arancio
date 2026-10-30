"""SHA-256 sidecars for session logs, with all file I/O supplied by Storage."""

from __future__ import annotations

import hashlib
from contextlib import closing
from pathlib import Path

from arancio.sessions.constants import SESSION_CHECKSUM_SUFFIX
from arancio.storage.manager import StorageManager


class SessionChecksum:
    """Read, verify and refresh a session log's checksum through Storage.

    Attributes:
        __storage_manager: the supplied byte and text I/O service.
    """

    def __init__(self, storage_manager: StorageManager) -> None:
        """Store the service used for log bytes and checksum text.

        Args:
            storage_manager: the generic file I/O service.
        """
        self.__storage_manager = storage_manager

    def write(self, log_path: Path) -> None:
        """Persist the current log digest in its sibling sidecar.

        Args:
            log_path: the session log whose bytes are hashed.
        """
        self.__storage_manager.write_file(
            self.__path(log_path), self.__digest(log_path)
        )

    def verify(self, log_path: Path) -> bool | None:
        """Compare current log bytes with the saved digest.

        Args:
            log_path: the session log to verify.

        Returns:
            True for a match, False for a mismatch, or None for a missing sidecar.
        """
        try:
            stored = self.__storage_manager.read_text(self.__path(log_path))
        except FileNotFoundError:
            return None
        return stored.strip() == self.__digest(log_path)

    def __digest(self, path: Path) -> str:
        """Hash Storage's bounded byte chunks and close the iterator.

        Args:
            path: the file to hash.

        Returns:
            The lowercase hexadecimal SHA-256 digest.
        """
        digest = hashlib.sha256()
        with closing(self.__storage_manager.iter_file_bytes(path)) as chunks:
            for chunk in chunks:
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def __path(log_path: Path) -> Path:
        """Derive a log's sibling checksum path.

        Args:
            log_path: the session log covered by the checksum.

        Returns:
            The log path with the checksum suffix appended.
        """
        return log_path.with_name(log_path.name + SESSION_CHECKSUM_SUFFIX)
