"""Write tool result parser."""

from typing import Any

from codo.parsers.tool_result.base import BaseToolResultParser
from codo.types.messages import ToolErrorMessage, ToolResultMessage


class WriteToolResultParser(BaseToolResultParser):
    """Parse write tool output into a tool result message."""

    @classmethod
    def parse(
        cls,
        call_id: str,
        output: Any,
        is_error: bool = False,
    ) -> ToolResultMessage:
        """Parse write tool output into a normalized result message.

        Args:
            call_id: identifier of the tool call this result answers.
            output: raw write tool output.
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

        file_path = output.get("file_path") or ""
        bytes_written = output.get("bytes_written") or 0
        total_lines = output.get("total_lines") or 0
        action = output.get("action") or "wrote"

        verb = {"created": "Created", "overwritten": "Overwrote"}.get(action, "Wrote")
        content = f"{verb} {file_path} ({bytes_written} bytes, {total_lines} lines)"

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
