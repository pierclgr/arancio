"""Read tool result parser."""

from typing import Any

from codo.parsers.tool_result.base import BaseToolResultParser
from codo.types.messages import ToolErrorMessage, ToolResultMessage


class ReadToolResultParser(BaseToolResultParser):
    """Parse read tool output into a tool result message."""

    @classmethod
    def parse(
        cls,
        call_id: str,
        output: Any,
        is_error: bool = False,
    ) -> ToolResultMessage:
        """Parse read tool output into a normalized result message.

        Args:
            call_id: identifier of the tool call this result answers.
            output: raw read tool output.
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

        content = "\n".join(parts)
        if is_error:
            return ToolErrorMessage(
                content=content,
                id=call_id,
                output=output,
            )
        return ToolResultMessage(
            content=content,
            id=call_id,
            output=output,
        )
