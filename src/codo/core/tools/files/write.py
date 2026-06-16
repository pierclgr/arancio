"""File write tool exposing a file create/overwrite primitive to LLM clients."""

from pathlib import Path
from typing import Type

from codo.core.parsers.tool_result.files.write import WriteFileToolResultParser
from codo.core.tools.base import BaseTool


class WriteFileTool(BaseTool):
    """Create a new file or overwrite an existing one on disk.

    Existing files are protected by a read-first guard: the tool refuses to
    overwrite a file that has not been read in this session, or whose mtime
    has drifted since the last recorded read. The guard is enforced via the
    shared :class:`~codo.core.tools.session.ToolSession` registry that
    :class:`~codo.core.tools.files.read.ReadFileTool` populates on every
    successful read.
    """

    _result_parser: Type[WriteFileToolResultParser] = WriteFileToolResultParser

    def _call(self, file_path: str, content: str) -> dict:
        """Write ``content`` to ``file_path`` and return write metadata.

        Args:
            file_path: absolute path of the file to create or overwrite.
            content: full file contents to write. No trailing newline is
                added implicitly.

        Returns:
            A dict with keys ``file_path`` (str, canonical absolute path),
            ``bytes_written`` (int, UTF-8 byte count of ``content``),
            ``action`` (``"created"`` or ``"overwritten"``) and
            ``total_lines`` (int, line count of ``content``).

        Raises:
            ValueError: when ``file_path`` is not absolute.
            FileNotFoundError: when the parent directory does not exist.
            IsADirectoryError: when ``file_path`` points to a directory.
            PermissionError: when ``file_path`` exists but was not read
                this session, or when its mtime has drifted since the
                last recorded read.
        """
        if not Path(file_path).is_absolute():
            raise ValueError(f"file_path must be absolute: {file_path}")

        path = Path(file_path)
        if path.exists() and path.is_dir():
            raise IsADirectoryError(f"Path is not a regular file: {file_path}")
        if not path.parent.exists():
            raise FileNotFoundError(f"Parent directory not found: {path.parent}")

        canonical = str(path.resolve()) if path.exists() else str(path.absolute())

        if path.exists():
            current_mtime = path.stat().st_mtime
            if not self._session.is_known(canonical):
                raise PermissionError(
                    f"File exists but was not read this session; "
                    f"read it first before overwriting: {file_path}"
                )
            if not self._session.is_fresh(canonical, current_mtime):
                raise PermissionError(
                    f"File has changed on disk since it was read; "
                    f"re-read before overwriting: {file_path}"
                )
            action = "overwritten"
        else:
            action = "created"

        encoded = content.encode("utf-8")
        path.write_bytes(encoded)

        self._session.record_read(path=canonical, mtime=path.stat().st_mtime)

        total_lines = len(content.splitlines()) if content else 0

        return {
            "file_path": canonical,
            "bytes_written": len(encoded),
            "action": action,
            "total_lines": total_lines,
        }
