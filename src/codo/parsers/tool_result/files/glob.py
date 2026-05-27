"""Glob tool result parser."""

from typing import Any

from codo.parsers.tool_result.base import BaseToolResultParser
from codo.types.messages import ToolErrorMessage, ToolResultMessage


class GlobToolResultParser(BaseToolResultParser):
    """Parse glob tool output into a tool result message."""

    @classmethod
    def parse(
        cls,
        call_id: str,
        output: Any,
        is_error: bool = False,
    ) -> ToolResultMessage:
        """Parse glob tool output into a normalized result message.

        Args:
            call_id: identifier of the tool call this result answers.
            output: raw glob tool output.
            is_error: whether the tool execution failed.

        Returns:
            A parsed tool result message.
        """
        if not isinstance(output, dict):
            if is_error:
                return ToolErrorMessage(
                    content=str(output),
                    id=call_id,
                    output=output,
                )
            return ToolResultMessage(
                content=str(output),
                id=call_id,
                output=output,
            )

        timed_out = output.get("timed_out")
        if timed_out or is_error:
            content = cls._format_content(output)
            if timed_out:
                content = f"{content}\n[timed out]" if content else "[timed out]"
            return ToolErrorMessage(
                content=content,
                id=call_id,
                output=output,
            )

        content = cls._format_content(output)
        return ToolResultMessage(
            content=content,
            id=call_id,
            output=output,
        )

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
