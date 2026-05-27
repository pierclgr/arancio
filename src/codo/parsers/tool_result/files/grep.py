"""Grep tool result parser."""

from typing import Any

from codo.parsers.tool_result.base import BaseToolResultParser
from codo.types.messages import ToolErrorMessage, ToolResultMessage


class GrepToolResultParser(BaseToolResultParser):
    """Parse grep tool output into a tool result message."""

    @classmethod
    def parse(
        cls,
        call_id: str,
        output: Any,
        is_error: bool = False,
    ) -> ToolResultMessage:
        """Parse grep tool output into a normalized result message.

        Args:
            call_id: identifier of the tool call this result answers.
            output: raw grep tool output.
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
            output: the grep tool result dict.

        Returns:
            A formatted string with matches and a summary footer.
        """
        output_mode = output.get("output_mode", "files_with_matches")
        matches: list = output.get("matches") or []
        total_matches = output.get("total_matches", len(matches))
        truncated = output.get("truncated")

        lines: list[str] = []
        if output_mode == "files_with_matches":
            lines.extend(str(m) for m in matches)
            noun = "file" if total_matches == 1 else "files"
            footer = f"[{total_matches} {noun} matched]"
        elif output_mode == "count":
            for m in matches:
                if isinstance(m, dict):
                    lines.append(f"{m['file']}:{m['count']}")
                else:
                    lines.append(str(m))
            noun = "file" if total_matches == 1 else "files"
            footer = f"[{total_matches} {noun} with matches]"
        else:
            for m in matches:
                if isinstance(m, dict):
                    prefix = (
                        f"{m['file']}-{m['line']}-"
                        if m.get("is_context")
                        else f"{m['file']}:{m['line']}:"
                    )
                    lines.append(f"{prefix}{m['content']}")
                else:
                    lines.append(str(m))
            noun = "match" if total_matches == 1 else "matches"
            unique_files = len({m.get("file") for m in matches if isinstance(m, dict)})
            footer = f"[{total_matches} {noun} across {unique_files} files]"

        parts: list[str] = []
        if lines:
            parts.append("\n".join(lines))

        if truncated:
            shown = len(matches)
            footer = f"[output truncated, showing first {shown} of {total_matches}]"

        parts.append(footer)
        return "\n".join(parts)
