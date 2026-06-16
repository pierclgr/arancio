"""Glob tool exposing filename-pattern search to LLM clients."""

import subprocess
from pathlib import Path
from typing import Type

from codo.core.parsers.tool_result.files.glob import GlobToolResultParser
from codo.core.tools.base import BaseTool
from codo.core.tools.session import ToolSession


class GlobTool(BaseTool):
    """Search filenames by glob pattern using ripgrep (``rg --files``).

    The tool runs two ``rg --files`` invocations and intersects their outputs: one
    enumerates the "allowed" file set respecting the instance's ``ignore_aware`` and
    ``hidden_aware`` flags, the other enumerates all files matching the user pattern
    with rg's ignore/hidden defaults disabled. The intersection yields pattern matches
    filtered by the requested ignore semantics. Results are resolved to absolute paths
    and sorted by modification time descending. This is a read-only tool -- it does not
    interact with ``ToolSession``.
    """

    _default_head_limit: int = 100
    _output_limit: int = 50_000
    _default_timeout: int = 60
    _result_parser: Type[GlobToolResultParser] = GlobToolResultParser

    def __init__(
        self,
        session: ToolSession | None = None,
        ignore_aware: bool = True,
        hidden_aware: bool = True,
    ) -> None:
        """Initialize the tool and configure ignore/hidden filter behavior.

        Args:
            session: optional :class:`ToolSession` override forwarded to the
                base class.
            ignore_aware: when ``True`` (default), files and directories matched
                by ``.gitignore``/``.ignore`` rules are excluded from results.
            hidden_aware: when ``True`` (default), hidden files and directories
                (those whose name starts with ``.``) are excluded from results.
        """
        super().__init__(session=session)
        self.ignore_aware = ignore_aware
        self.hidden_aware = hidden_aware

    def _call(
        self,
        pattern: str,
        path: str | None = None,
        head_limit: int | None = None,
    ) -> dict:
        """Run ripgrep file enumeration and return structured match results.

        Args:
            pattern: glob pattern to match against filenames (e.g. ``**/*.py``).
            path: absolute path to a directory to search. Defaults to the current
                working directory.
            head_limit: maximum number of paths to return. Defaults to
                :attr:`_default_head_limit`.

        Returns:
            A dict with keys ``matches`` (list of absolute path strings),
            ``total_matches`` (int), ``truncated`` (bool), ``timed_out`` (bool),
            ``exit_code`` (int), ``search_path`` (str) and ``pattern`` (str).
        """
        search_path = path or "."
        resolved_search_path = str(Path(search_path).resolve())

        allowed, allowed_exit, allowed_timed_out = self._run_rg(
            self._allowed_args(search_path),
        )
        matched, matched_exit, matched_timed_out = self._run_rg(
            self._pattern_args(search_path, pattern),
        )

        timed_out = allowed_timed_out or matched_timed_out
        exit_code = -1 if timed_out else matched_exit

        if matched_exit == 1 and not timed_out:
            return {
                "matches": [],
                "total_matches": 0,
                "truncated": False,
                "timed_out": False,
                "exit_code": 1,
                "search_path": resolved_search_path,
                "pattern": pattern,
            }

        intersection = matched & allowed if allowed_exit == 0 else set()
        resolved = [Path(line).resolve() for line in intersection]
        sorted_paths = sorted(resolved, key=self._mtime_desc_key, reverse=True)
        total_matches = len(sorted_paths)

        effective_limit = head_limit or self._default_head_limit
        truncated = total_matches > effective_limit
        if truncated:
            sorted_paths = sorted_paths[:effective_limit]

        return {
            "matches": [str(p) for p in sorted_paths],
            "total_matches": total_matches,
            "truncated": truncated,
            "timed_out": timed_out,
            "exit_code": exit_code,
            "search_path": resolved_search_path,
            "pattern": pattern,
        }

    def _allowed_args(self, search_path: str) -> list[str]:
        """Build rg args for the "allowed" file enumeration pass.

        Args:
            search_path: directory to search.

        Returns:
            The argv list to invoke rg with.
        """
        args = ["rg", "--files", "--no-heading"]
        if not self.ignore_aware:
            args.append("--no-ignore")
        if not self.hidden_aware:
            args.append("--hidden")
        args.extend(["--", search_path])
        return args

    @staticmethod
    def _pattern_args(search_path: str, pattern: str) -> list[str]:
        """Build rg args for the pattern-match pass (ignore/hidden disabled).

        Args:
            search_path: directory to search.
            pattern: glob pattern to filter against.

        Returns:
            The argv list to invoke rg with.
        """
        return [
            "rg",
            "--files",
            "--no-heading",
            "--no-ignore",
            "--hidden",
            "-g",
            pattern,
            "--",
            search_path,
        ]

    def _run_rg(self, args: list[str]) -> tuple[set[str], int, bool]:
        """Run an rg invocation and normalize the result.

        Args:
            args: the rg argv list to invoke.

        Returns:
            A tuple ``(files, exit_code, timed_out)`` where ``files`` is the set
            of non-empty stdout lines.

        Raises:
            RuntimeError: when rg exits with code 2 or stderr is present on a
                non-zero exit.
        """
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

        files = {line for line in stdout.splitlines() if line}
        return files, exit_code, timed_out

    @staticmethod
    def _mtime_desc_key(p: Path) -> float:
        """Return mtime for sorting, falling back to 0 for missing files.

        Args:
            p: a resolved path.

        Returns:
            The path's mtime, or ``0.0`` if the file was removed since enumeration.
        """
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    @staticmethod
    def _decode(stream: bytes | str | None) -> str:
        """Normalize a captured stream to a string.

        Args:
            stream: the raw stream contents, possibly ``None`` or bytes when raised
                from a ``TimeoutExpired`` exception.

        Returns:
            The stream decoded as UTF-8 with replacement on errors, or an empty
            string when ``stream`` is ``None``.
        """
        if stream is None:
            return ""
        if isinstance(stream, bytes):
            return stream.decode(errors="replace")
        return stream
