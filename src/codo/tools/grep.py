"""Grep tool exposing ripgrep content search to LLM clients."""

import re
import subprocess
from pathlib import Path
from typing import Type

from codo.parsers.tool_result.grep import GrepToolResultParser
from codo.tools.base import BaseTool


class GrepTool(BaseTool):
    """Search file contents using ripgrep (``rg``) with structured output modes.

    The tool shells out to ``rg`` with the requested flags, parses its stdout, and
    returns a structured dict with matches, counts, and truncation status. This is a
    read-only tool -- it does not interact with ``ToolSession``.
    """

    _default_head_limit: int = 100
    _output_limit: int = 50_000
    _default_timeout: int = 60
    _result_parser: Type[GrepToolResultParser] = GrepToolResultParser

    def _call(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        file_type: str | None = None,
        output_mode: str = "files_with_matches",
        i: bool = False,
        n: bool = True,
        A: int | None = None,
        B: int | None = None,
        C: int | None = None,
        head_limit: int | None = None,
        multiline: bool = False,
    ) -> dict:
        """Run ripgrep and return structured match results.

        Args:
            pattern: regex pattern to search for.
            path: absolute path to a file or directory to search.
                Defaults to the current working directory.
            glob: filename pattern filter (e.g. ``*.py``).
            file_type: language type filter using rg's built-in type
                map (e.g. ``py``, ``js``).
            output_mode: one of ``files_with_matches`` (default),
                ``content`` or ``count``.
            i: case insensitive search when ``True``.
            n: show line numbers in content mode when ``True``.
            A: lines to show after each match.
            B: lines to show before each match.
            C: lines to show before and after each match.
            head_limit: maximum number of matches to return. Defaults
                to :attr:`_default_head_limit`.
            multiline: enable multi-line matching when ``True``.

        Returns:
            A dict with keys ``matches`` (list), ``total_matches``
            (int), ``truncated`` (bool), ``timed_out`` (bool),
            ``exit_code`` (int) and ``output_mode`` (str).

        Raises:
            RuntimeError: when rg exits with code 2 or stderr is
                present on a non-zero exit.
        """
        args = ["rg", "--no-heading"]

        if output_mode == "files_with_matches":
            args.append("--files-with-matches")
        elif output_mode == "count":
            args.append("--count")
        else:
            if n:
                args.append("--line-number")

        if glob:
            args.extend(["-g", glob])
        if file_type:
            args.extend(["-t", file_type])
        if i:
            args.append("-i")
        if multiline:
            args.append("--multiline")
        if A is not None:
            args.extend(["-A", str(A)])
        if B is not None:
            args.extend(["-B", str(B)])
        if C is not None:
            args.extend(["-C", str(C)])

        args.append("--")
        args.append(pattern)
        args.append(path or ".")

        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self._default_timeout,
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

        if exit_code == 2 or (stderr and exit_code not in (0, 1)):
            raise RuntimeError(stderr.strip() or f"rg exited with code {exit_code}")

        if exit_code == 1 and not timed_out:
            return {
                "matches": [],
                "total_matches": 0,
                "truncated": False,
                "timed_out": False,
                "exit_code": 1,
                "output_mode": output_mode,
            }

        lines = stdout.splitlines()
        total_matches = len(lines)
        effective_limit = head_limit or self._default_head_limit

        matches = self._parse_lines(lines, output_mode)
        truncated = len(matches) > effective_limit
        if truncated:
            matches = matches[:effective_limit]

        return {
            "matches": matches,
            "total_matches": total_matches,
            "truncated": truncated,
            "timed_out": timed_out,
            "exit_code": exit_code,
            "output_mode": output_mode,
        }

    @staticmethod
    def _parse_lines(lines: list[str], output_mode: str) -> list:
        """Parse rg stdout lines into structured match items.

        Args:
            lines: raw stdout lines from rg.
            output_mode: the output mode that was requested.

        Returns:
            A list of match items whose shape depends on the mode.
        """
        if output_mode == "files_with_matches":
            return [str(Path(line).resolve()) for line in lines if line]

        if output_mode == "count":
            results: list = []
            for line in lines:
                if not line:
                    continue
                parts = line.rsplit(":", 1)
                if len(parts) == 2 and parts[1].strip().isdigit():
                    results.append(
                        {
                            "file": str(Path(parts[0]).resolve()),
                            "count": int(parts[1]),
                        }
                    )
                else:
                    results.append(line)
            return results

        # content mode — use regex because context separator '-' can
        # appear in absolute file paths; greedy match finds the last
        # '-<digits>-' which is the correct lineno marker from rg
        _CONTEXT_RE = re.compile(r"^(.+)-(\d+)-(.*)$")
        _MATCH_RE = re.compile(r"^(.+):(\d+):(.*)$")
        results = []
        for line in lines:
            if not line:
                continue
            m = _MATCH_RE.match(line)
            if m:
                results.append(
                    {
                        "file": str(Path(m.group(1)).resolve()),
                        "line": int(m.group(2)),
                        "content": m.group(3),
                        "is_context": False,
                    }
                )
                continue
            m = _CONTEXT_RE.match(line)
            if m:
                results.append(
                    {
                        "file": str(Path(m.group(1)).resolve()),
                        "line": int(m.group(2)),
                        "content": m.group(3),
                        "is_context": True,
                    }
                )
                continue
            results.append(line)
        return results

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
