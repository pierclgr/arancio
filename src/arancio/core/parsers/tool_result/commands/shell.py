"""Shell command tool result parser."""

from arancio.core.parsers.tool_result.base import BaseToolResultParser


class ShellCommandToolResultParser(BaseToolResultParser):
    """Parse shell command output into a tool result message."""

    @classmethod
    def _render(cls, output: dict) -> str:
        """Render shell command output with status footers.

        Args:
            output: the structured shell command output.

        Returns:
            The combined stdout/stderr with timeout, exit-code and truncation
            annotations.
        """
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

        return "\n".join(parts)

    @classmethod
    def _failed(cls, output: dict) -> bool:
        """Report failure on a timeout or a non-zero exit code.

        Args:
            output: the structured shell command output.

        Returns:
            True when the command timed out or exited with a non-zero code.
        """
        return bool(output.get("timed_out")) or output.get("exit_code") not in (None, 0)
