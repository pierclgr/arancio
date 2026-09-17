"""Shared in-memory state for cross-tool coordination."""


class ToolSession:
    """Track files seen by read tools so write tools can refuse blind overwrites.

    A single :class:`ToolSession` instance is shared across tools in a run, so a write
    tool can detect that an existing file was never read this session or that its mtime
    drifted since it was read. The session does not persist across processes; each agent
    run starts (or resets) it.
    """

    def __init__(self) -> None:
        """Initialize an empty session."""
        self._reads: dict[str, float] = {}

    def record_read(self, path: str, mtime: float) -> None:
        """Record that ``path`` was read at the given ``mtime``.

        Args:
            path: canonical absolute path of the file that was read.
            mtime: the file's modification timestamp at the time of the read.
        """
        self._reads[path] = mtime

    def is_known(self, path: str) -> bool:
        """Return whether ``path`` has been read at least once this session.

        Args:
            path: canonical absolute path of the file to check.

        Returns:
            ``True`` when the path has a recorded read, ``False`` otherwise.
        """
        return path in self._reads

    def is_fresh(self, path: str, current_mtime: float) -> bool:
        """Return whether the recorded read of ``path`` is still up to date.

        Args:
            path: canonical absolute path of the file to check.
            current_mtime: the file's current modification timestamp.

        Returns:
            ``True`` when the path was read and its recorded mtime matches
            ``current_mtime``. ``False`` when the path is unknown or its
            mtime has drifted since the recorded read.
        """
        recorded = self._reads.get(path)
        return recorded is not None and recorded == current_mtime

    def snapshot(self) -> dict[str, float]:
        """Return every recorded read as a path-to-modification-time mapping.

        The mapping is a copy, so a caller reading it never observes later
        reads and cannot mutate the session through it.

        Returns:
            A copy of the recorded reads.
        """
        return dict(self._reads)

    def clear(self) -> None:
        """Drop all recorded reads.

        A new or replaced chat starts from a clean slate, since the module-level
        :data:`shared_session` lives for the whole process.
        """
        self._reads.clear()


shared_session: ToolSession = ToolSession()
