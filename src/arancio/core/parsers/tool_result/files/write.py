"""Write file tool result parser."""

from arancio.core.parsers.tool_result.base import BaseToolResultParser


class WriteFileToolResultParser(BaseToolResultParser):
    """Parse write file tool output into a tool result message."""

    @classmethod
    def _render(cls, output: dict) -> str:
        """Render write file output into a one-line summary.

        Args:
            output: the structured write file tool output.

        Returns:
            A summary naming the action, path, bytes written and line count.
        """
        file_path = output.get("file_path") or ""
        bytes_written = output.get("bytes_written") or 0
        total_lines = output.get("total_lines") or 0
        action = output.get("action") or "wrote"

        verb = {"created": "Created", "overwritten": "Overwrote"}.get(action, "Wrote")
        return f"{verb} {file_path} ({bytes_written} bytes, {total_lines} lines)"
