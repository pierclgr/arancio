"""Edit tool result parser."""

from typing import Any

from codo.parsers.tool_result.base import BaseToolResultParser
from codo.types.messages import ToolErrorMessage, ToolResultMessage


class EditToolResultParser(BaseToolResultParser):
    """Parse edit tool output into a tool result message."""

    @classmethod
    def parse(
        cls,
        call_id: str,
        output: Any,
        is_error: bool = False,
    ) -> ToolResultMessage:
        """Parse edit tool output into a normalized result message.

        Args:
            call_id: identifier of the tool call this result answers.
            output: raw edit tool output.
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
        replacements = output.get("replacements") or 0
        bytes_before = output.get("bytes_before") or 0
        bytes_after = output.get("bytes_after") or 0

        noun = "replacement" if replacements == 1 else "replacements"
        content = (
            f"Edited {file_path} "
            f"({replacements} {noun}, {bytes_before} → {bytes_after} bytes)"
        )

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
