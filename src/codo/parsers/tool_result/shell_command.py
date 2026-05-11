"""Shell command tool result parser."""

from typing import Any

from codo.parsers.tool_result.base import BaseToolResultParser
from codo.types.messages import ToolErrorMessage, ToolResultMessage


class ShellCommandToolResultParser(BaseToolResultParser):
    """Parse shell command output into a tool result message."""

    @classmethod
    def parse(
        cls,
        call_id: str,
        output: Any,
        is_error: bool = False,
    ) -> ToolResultMessage:
        """Parse shell command output into a normalized result message.

        Args:
            call_id: identifier of the tool call this result answers.
            output: raw shell command output.
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

        stdout = output.get("stdout") or ""
        stderr = output.get("stderr") or ""
        exit_code = output.get("exit_code")
        timed_out = output.get("timed_out")
        truncated = output.get("truncated")

        parts = []
        if stdout:
            parts.append(stdout.rstrip())
        if stderr:
            parts.append(stderr.rstrip())
        if timed_out:
            parts.append("[timed out]")
        if exit_code not in (None, 0):
            parts.append(f"[exit code {exit_code}]")
        if truncated:
            parts.append("[output truncated]")

        content = "\n".join(parts)
        if is_error or bool(timed_out) or exit_code not in (None, 0):
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
