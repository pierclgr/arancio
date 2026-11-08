"""File read tool exposing a text-file reader to LLM clients."""

from pathlib import Path
from typing import Type

from dynamic_markdown.types.files.base import DynamicMarkdownFile

from arancio.core.parsers.tool_result.files.read import ReadFileToolResultParser
from arancio.core.tools.base import Tool


class ReadFileTool(Tool):
    """Read a text file from disk with ``cat -n``-style line numbering.

    Select and cap raw source lines before expanding complete dynamic Markdown tags in
    ``.md`` files by default. Plain-text and raw Markdown prefixes retain source line
    numbers; expanded Markdown prefixes number the excerpt from one, with source bounds
    reported separately. Expanded output has no line-count cap. Every output line still
    has a per-line character cap.
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
        raw_markdown: bool = False,
    ) -> dict:
        """Read a file slice and return its line-numbered contents.

        Args:
            file_path: absolute path to the file to read.
            offset: 1-indexed source line to start paging from. Defaults to one.
                An offset beyond the last source line reads from the beginning.
            limit: source-line page size, capped at ``_default_limit``.
            raw_markdown: whether to return raw ``.md`` source with source line
                numbers. Defaults to False, expanding dynamic Markdown tags.
                Has no effect on other file types.

        Returns:
            A dict containing ``file_path``, numbered ``content``, ``start_line``,
            ``end_line``, ``total_lines`` and ``truncated_lines``. Plain-text
            and raw Markdown bounds refer to source positions. Expanded Markdown
            bounds describe the excerpt; ``source_start_line``, ``source_end_line`` and
            ``source_total_lines`` describe its raw source selection. Empty
            selections have zero start/end bounds.

        Raises:
            ValueError: when ``file_path`` is not absolute, when paging arguments
                are not positive, or when a ``.md`` file's
                dynamic-markdown expansion fails (e.g. an unresolvable
                ``<field>`` tag or an ``<include>`` cycle).
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

        text: str = path.read_text(encoding="utf-8", errors="replace")
        all_lines: list[str] = text.splitlines(keepends=True)
        total_lines: int = len(all_lines)

        if offset > total_lines:
            offset = 1
            limit = None
        effective_limit: int = min(limit or self._default_limit, self._default_limit)

        start_idx = offset - 1
        end_idx = min(start_idx + effective_limit, total_lines)
        selected: list[str] = (
            all_lines[start_idx:end_idx] if start_idx < total_lines else []
        )
        source_start: int = start_idx + 1 if selected else 0
        source_end: int = start_idx + len(selected) if selected else 0
        source_total: int = total_lines
        selected_text: str = "".join(selected)

        should_expand: bool = path.suffix == ".md" and not raw_markdown
        if should_expand:
            selected_text = DynamicMarkdownFile(path, content=selected_text).content
            start_idx = 0
        selected = selected_text.splitlines()
        if should_expand:
            total_lines = len(selected)

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

        self._tool_session.record_read(path=canonical, mtime=mtime)

        output: dict = {
            "file_path": canonical,
            "content": formatted,
            "start_line": start_idx + 1 if selected else 0,
            "end_line": start_idx + len(selected) if selected else 0,
            "total_lines": total_lines,
            "truncated_lines": truncated_lines,
        }
        if should_expand:
            output.update(
                source_start_line=source_start,
                source_end_line=source_end,
                source_total_lines=source_total,
            )
        return output
