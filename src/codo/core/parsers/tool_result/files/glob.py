"""Glob tool result parser."""

from codo.core.parsers.tool_result.base import BaseToolResultParser


class GlobToolResultParser(BaseToolResultParser):
    """Parse glob tool output into a tool result message."""

    @classmethod
    def _render(cls, output: dict) -> str:
        """Render the path list, appending a timeout marker when needed.

        Args:
            output: the structured glob tool output.

        Returns:
            The formatted path block, with a ``[timed out]`` suffix on timeout.
        """
        display_text = cls._format_content(output)
        if output.get("timed_out"):
            display_text = (
                f"{display_text}\n[timed out]" if display_text else "[timed out]"
            )
        return display_text

    @classmethod
    def _failed(cls, output: dict) -> bool:
        """Report failure when the listing timed out.

        Args:
            output: the structured glob tool output.

        Returns:
            True when the listing timed out.
        """
        return bool(output.get("timed_out"))

    @classmethod
    def _format_content(cls, output: dict) -> str:
        """Format the match list into a human-readable block with footer.

        Args:
            output: the glob tool result dict.

        Returns:
            A formatted string with absolute paths and a summary footer.
        """
        matches: list = output.get("matches") or []
        total_matches = output.get("total_matches", len(matches))
        truncated = output.get("truncated")

        if total_matches == 0:
            return "[no files matched]"

        noun = "file" if total_matches == 1 else "files"
        if truncated:
            shown = len(matches)
            footer = f"[output truncated, showing first {shown} of {total_matches}]"
        else:
            footer = f"[{total_matches} {noun} matched, sorted by mtime]"

        parts: list[str] = []
        if matches:
            parts.append("\n".join(str(m) for m in matches))
        parts.append(footer)
        return "\n".join(parts)
