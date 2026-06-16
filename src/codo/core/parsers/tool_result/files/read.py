"""Read file tool result parser."""

from codo.core.parsers.tool_result.base import BaseToolResultParser


class ReadFileToolResultParser(BaseToolResultParser):
    """Parse read file tool output into a tool result message."""

    @classmethod
    def _render(cls, output: dict) -> str:
        """Render read file output into a content block with a status footer.

        Args:
            output: the structured read file tool output.

        Returns:
            The file content followed by line-range and truncation annotations.
        """
        content_block = output.get("content") or ""
        total_lines = output.get("total_lines") or 0
        start_line = output.get("start_line") or 0
        end_line = output.get("end_line") or 0
        truncated_lines = output.get("truncated_lines") or 0

        parts = []
        if content_block:
            parts.append(content_block)
        if total_lines == 0:
            parts.append("[empty file]")
        elif start_line == 0:
            parts.append(f"[no lines returned, file has {total_lines} lines]")
        else:
            parts.append(f"[lines {start_line}-{end_line} of {total_lines}]")
        if truncated_lines:
            parts.append(f"[{truncated_lines} long lines truncated]")

        return "\n".join(parts)
