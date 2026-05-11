"""File edit tool exposing an exact-substring file replacement primitive."""

from pathlib import Path
from typing import Type

from codo.parsers.tool_result.files.edit import EditFileToolResultParser
from codo.tools.base import BaseTool


class EditFileTool(BaseTool):
    """Replace an exact substring in an existing UTF-8 file.

    The tool refuses to edit a file that was not read in this session or whose mtime has
    drifted since the last recorded read, using the shared
    :class:`~codo.tools.session.ToolSession` registry that other tools populate. The
    match is strictly byte-exact and must be unique unless ``replace_all`` is set.
    """

    _result_parser: Type[EditFileToolResultParser] = EditFileToolResultParser

    def _call(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> dict:
        """Replace ``old_string`` with ``new_string`` in ``file_path``.

        Args:
            file_path: absolute path of the file to edit.
            old_string: the verbatim substring to find. Must be non-empty
                and, unless ``replace_all`` is true, must be unique in the
                file.
            new_string: the substring that replaces ``old_string``. May be
                empty (deletes the matched snippet) but must differ from
                ``old_string``.
            replace_all: when true, replace every occurrence; when false
                (default), require the match to be unique.

        Returns:
            A dict with keys ``file_path`` (str, canonical absolute path),
            ``replacements`` (int, occurrences replaced), ``bytes_before``
            (int, UTF-8 byte size before the edit), ``bytes_after`` (int,
            UTF-8 byte size after the edit) and ``action`` (always
            ``"edited"``).

        Raises:
            ValueError: when ``file_path`` is not absolute, when
                ``old_string`` is empty, when ``old_string`` equals
                ``new_string``, when ``old_string`` is not found, or
                when ``old_string`` is not unique and ``replace_all`` is
                false.
            FileNotFoundError: when ``file_path`` does not exist.
            IsADirectoryError: when ``file_path`` points to a directory.
            PermissionError: when ``file_path`` was not read this session
                or when its mtime has drifted since the last recorded
                read.
        """
        if not Path(file_path).is_absolute():
            raise ValueError(f"file_path must be absolute: {file_path}")
        if not old_string:
            raise ValueError("old_string must not be empty")
        if old_string == new_string:
            raise ValueError("old_string and new_string are identical; nothing to do")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise IsADirectoryError(f"Path is not a regular file: {file_path}")

        canonical = str(path.resolve())
        current_mtime = path.stat().st_mtime

        if not self._session.is_known(canonical):
            raise PermissionError(
                f"File was not read this session; "
                f"read it first before editing: {file_path}"
            )
        if not self._session.is_fresh(canonical, current_mtime):
            raise PermissionError(
                f"File has changed on disk since it was read; "
                f"re-read before editing: {file_path}"
            )

        text = path.read_text(encoding="utf-8")
        occurrences = text.count(old_string)

        if occurrences == 0:
            raise ValueError("old_string not found in file")
        if occurrences > 1 and not replace_all:
            raise ValueError(
                f"old_string is not unique (occurs {occurrences} times); "
                f"use replace_all=True or expand context"
            )

        replacements = occurrences if replace_all else 1
        new_text = text.replace(
            old_string,
            new_string,
            -1 if replace_all else 1,
        )

        bytes_before = len(text.encode("utf-8"))
        encoded = new_text.encode("utf-8")
        path.write_bytes(encoded)

        self._session.record_read(path=canonical, mtime=path.stat().st_mtime)

        return {
            "file_path": canonical,
            "replacements": replacements,
            "bytes_before": bytes_before,
            "bytes_after": len(encoded),
            "action": "edited",
        }
