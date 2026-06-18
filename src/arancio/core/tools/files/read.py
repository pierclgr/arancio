"""File read tool exposing a text-file reader to LLM clients."""

from pathlib import Path
from typing import Type

from arancio.core.parsers.tool_result.files.read import ReadFileToolResultParser
from arancio.core.tools.base import BaseTool


class ReadFileTool(BaseTool):
    """Read a text file from disk with ``cat -n``-style line numbering.

    The tool reads a UTF-8 file, optionally slicing it with ``offset`` and ``limit``,
    and returns each line prefixed with its 1-indexed line number and a tab so the
    output can feed a downstream edit tool. Long lines are truncated past a per-line
    character cap to keep the model context bounded.
    """

    _default_limit: int = 2000
    _max_line_chars: int = 2000
    _line_truncation_marker: str = "… [line truncated]"
    _result_parser: Type[ReadFileToolResultParser] = ReadFileToolResultParser

    def _call(
        self,
        file_path: str,
        offset: int = 1,
        limit: int | None = None,
    ) -> dict:
        """Read a file slice and return its line-numbered contents.

        Args:
            file_path: absolute path to the file to read.
            offset: 1-indexed line number to start reading from. Defaults
                to ``1``.
            limit: maximum number of lines to read. Defaults to
                ``_default_limit`` and is capped at ``_default_limit``.

        Returns:
            A dict with keys ``content`` (str, ``cat -n``-formatted slice),
            ``start_line`` (int, 1-indexed first returned line or ``0``),
            ``end_line`` (int, 1-indexed last returned line or ``0``),
            ``total_lines`` (int, total line count in the file) and
            ``truncated_lines`` (int, number of returned lines truncated to
            the per-line character cap).

        Raises:
            ValueError: when ``file_path`` is not absolute or when
                ``offset`` or ``limit`` are not positive.
            FileNotFoundError: when ``file_path`` does not exist.
            IsADirectoryError: when ``file_path`` points to a directory.
        """
        if not Path(file_path).is_absolute():
            raise ValueError(f"file_path must be absolute: {file_path}")
        if offset < 1:
            raise ValueError(f"offset must be >= 1: {offset}")
        if limit is not None and limit < 1:
            raise ValueError(f"limit must be >= 1: {limit}")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise IsADirectoryError(f"Path is not a regular file: {file_path}")

        canonical = str(path.resolve())
        mtime = path.stat().st_mtime

        effective_limit = min(limit or self._default_limit, self._default_limit)

        text = path.read_text(encoding="utf-8", errors="replace")
        all_lines = text.splitlines()
        total_lines = len(all_lines)

        start_idx = offset - 1
        end_idx = min(start_idx + effective_limit, total_lines)
        selected = all_lines[start_idx:end_idx] if start_idx < total_lines else []

        truncated_lines = 0
        capped_lines = []
        for line in selected:
            if len(line) > self._max_line_chars:
                truncated_lines += 1
                capped_lines.append(
                    line[: self._max_line_chars] + self._line_truncation_marker
                )
            else:
                capped_lines.append(line)

        formatted = "\n".join(
            f"{start_idx + 1 + i:>6}\t{line}" for i, line in enumerate(capped_lines)
        )

        self._session.record_read(path=canonical, mtime=mtime)

        return {
            "content": formatted,
            "start_line": start_idx + 1 if selected else 0,
            "end_line": start_idx + len(selected) if selected else 0,
            "total_lines": total_lines,
            "truncated_lines": truncated_lines,
        }
