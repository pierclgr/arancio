"""Bash command tool exposing a shell command runner to LLM clients."""

import subprocess
from typing import Type

from codo.parsers.tool_result.shell_command import ShellCommandToolResultParser
from codo.tools.base import BaseTool


class BashCommandTool(BaseTool):
    """Run a shell command via ``/bin/sh -c`` and return its output.

    The tool captures ``stdout`` and ``stderr`` separately, returns the process exit
    code and reports whether the run timed out or produced truncated output.
    """

    _default_timeout: int = 120
    _max_timeout: int = 600
    _output_limit: int = 30_000
    _result_parser: Type[ShellCommandToolResultParser] = ShellCommandToolResultParser

    def _call(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
    ) -> dict:
        """Run a shell command and return its raw captured output.

        Args:
            command: the shell command line to execute.
            timeout: maximum runtime in seconds. Defaults to ``120`` and
                is capped at ``600``.
            cwd: absolute working directory for the command. Defaults
                to the current process working directory.

        Returns:
            A dict with keys ``stdout`` (str), ``stderr`` (str),
            ``exit_code`` (int), ``timed_out`` (bool) and ``truncated``
            (bool). On timeout ``exit_code`` is ``-1`` and any partial
            output captured before the timeout is included.
        """
        effective_timeout = min(timeout or self._default_timeout, self._max_timeout)

        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            exit_code = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout = self._decode(exc.stdout)
            stderr = self._decode(exc.stderr)
            exit_code = -1
            timed_out = True

        stdout, stdout_truncated = self._truncate(stdout)
        stderr, stderr_truncated = self._truncate(stderr)

        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "truncated": stdout_truncated or stderr_truncated,
        }

    @staticmethod
    def _decode(stream: bytes | str | None) -> str:
        """Normalize a captured stream to a string.

        Args:
            stream: the raw stream contents, possibly ``None`` or bytes
                when raised from a ``TimeoutExpired`` exception.

        Returns:
            The stream decoded as UTF-8 with replacement on errors, or
            an empty string when ``stream`` is ``None``.
        """
        if stream is None:
            return ""
        if isinstance(stream, bytes):
            return stream.decode(errors="replace")
        return stream

    @classmethod
    def _truncate(cls, output: str) -> tuple[str, bool]:
        """Cap a stream at the output limit, appending a truncation marker.

        Args:
            output: the captured stream contents.

        Returns:
            A tuple of the (possibly truncated) string and a flag
            indicating whether truncation occurred.
        """
        if len(output) <= cls._output_limit:
            return output, False
        dropped = len(output) - cls._output_limit
        return (
            output[: cls._output_limit] + f"\n… [{dropped} chars truncated]",
            True,
        )
