"""Edit file tool result parser."""

from codo.core.parsers.tool_result.base import BaseToolResultParser


class EditFileToolResultParser(BaseToolResultParser):
    """Parse edit file tool output into a tool result message."""

    @classmethod
    def _render(cls, output: dict) -> str:
        """Render edit file output into a summary followed by the diff.

        Args:
            output: the structured edit file tool output.

        Returns:
            An edit summary, optionally followed by the unified diff.
        """
        file_path = output.get("file_path") or ""
        replacements = output.get("replacements") or 0
        bytes_before = output.get("bytes_before") or 0
        bytes_after = output.get("bytes_after") or 0

        noun = "replacement" if replacements == 1 else "replacements"
        display_text = (
            f"Edited {file_path} "
            f"({replacements} {noun}, {bytes_before} → {bytes_after} bytes)"
        )

        diff = output.get("diff") or ""
        if diff:
            display_text = f"{display_text}\n{diff}"

        return display_text
