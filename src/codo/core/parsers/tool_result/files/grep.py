"""Grep tool result parser."""

from codo.core.parsers.tool_result.base import BaseToolResultParser


class GrepToolResultParser(BaseToolResultParser):
    """Parse grep tool output into a tool result message."""

    @classmethod
    def _render(cls, output: dict) -> str:
        """Render the match list, appending a timeout marker when needed.

        Args:
            output: the structured grep tool output.

        Returns:
            The formatted match block, with a ``[timed out]`` suffix on timeout.
        """
        display_text = cls._format_content(output)
        if output.get("timed_out"):
            display_text = (
                f"{display_text}\n[timed out]" if display_text else "[timed out]"
            )
        return display_text

    @classmethod
    def _failed(cls, output: dict) -> bool:
        """Report failure when the search timed out.

        Args:
            output: the structured grep tool output.

        Returns:
            True when the search timed out.
        """
        return bool(output.get("timed_out"))

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
