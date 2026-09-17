"""Tests for the concrete tools and the read-first guard they share.

The write and edit tools refuse to touch a file the session has not read, which is the
whole reason :class:`~arancio.core.tools.session.ToolSession` exists. Those tests drive
the real tools against real files under ``tmp_path`` and the real process-wide guard,
which the autouse fixture in ``conftest`` empties between tests.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest
from ddgs.exceptions import TimeoutException
from fakes import ScriptedClient

import arancio.core.tools.web.fetch as fetch_module
import arancio.core.tools.web.search as search_module
from arancio.core.messages import (
    AssistantChunkMessage,
    AssistantMessage,
    ToolErrorMessage,
)
from arancio.core.tools.commands.shell import ShellCommandTool
from arancio.core.tools.files.edit import EditFileTool
from arancio.core.tools.files.read import ReadFileTool
from arancio.core.tools.files.write import WriteFileTool
from arancio.core.tools.session import shared_session
from arancio.core.tools.web.fetch import FetchWebTool
from arancio.core.tools.web.search import SearchWebTool


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    """Return a three-line file to read, write and edit.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The path of the written file.
    """
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n")
    return path


def _read(path: Path) -> dict:
    """Read a file through the real tool, arming the shared guard.

    Args:
        path: the file to read.

    Returns:
        The tool's result payload.
    """
    return ReadFileTool()._call(file_path=str(path))


def test_read_numbers_every_line(sample: Path) -> None:
    """A whole-file read is formatted like ``cat -n``."""
    result = _read(sample)

    assert result["content"] == "     1\talpha\n     2\tbeta\n     3\tgamma"
    assert result["start_line"] == 1
    assert result["end_line"] == 3
    assert result["total_lines"] == 3
    assert result["file_path"] == str(sample.resolve())


def test_read_slices_with_offset_and_limit(sample: Path) -> None:
    """The window keeps the file's own line numbers, not the slice's."""
    result = ReadFileTool()._call(file_path=str(sample), offset=2, limit=1)

    assert result["content"] == "     2\tbeta"
    assert result["start_line"] == 2
    assert result["end_line"] == 2
    assert result["total_lines"] == 3


def test_read_past_the_end_returns_an_empty_slice(sample: Path) -> None:
    """An offset beyond the file reports no lines rather than failing."""
    result = ReadFileTool()._call(file_path=str(sample), offset=99)

    assert result["content"] == ""
    assert result["start_line"] == 0
    assert result["end_line"] == 0
    assert result["total_lines"] == 3


def test_read_caps_a_very_long_line(tmp_path: Path) -> None:
    """A line past the per-line cap is cut and counted, keeping context bounded."""
    path = tmp_path / "long.txt"
    path.write_text("x" * 2500)

    result = ReadFileTool()._call(file_path=str(path))

    assert result["truncated_lines"] == 1
    assert result["content"].endswith("… [line truncated]")


def test_read_arms_the_shared_guard(sample: Path) -> None:
    """A successful read is what later lets a write through."""
    _read(sample)

    assert shared_session.is_known(str(sample.resolve()))
    assert shared_session.is_fresh(str(sample.resolve()), sample.stat().st_mtime)


def test_read_rejects_a_relative_path(tmp_path: Path) -> None:
    """Every file tool works in absolute paths only."""
    with pytest.raises(ValueError):
        ReadFileTool()._call(file_path="sample.txt")


def test_read_rejects_a_missing_file(tmp_path: Path) -> None:
    """A missing file is a plain ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        ReadFileTool()._call(file_path=str(tmp_path / "nope.txt"))


def test_read_rejects_a_directory(tmp_path: Path) -> None:
    """A directory is not a readable file."""
    with pytest.raises(IsADirectoryError):
        ReadFileTool()._call(file_path=str(tmp_path))


def test_write_creates_a_new_file_without_a_prior_read(tmp_path: Path) -> None:
    """The guard protects existing content, so a new file needs no read."""
    path = tmp_path / "nested" / "new.txt"

    result = WriteFileTool()._call(file_path=str(path), content="hello\n")

    assert result["action"] == "created"
    assert result["bytes_written"] == 6
    assert path.read_text() == "hello\n"


def test_write_refuses_to_overwrite_a_file_it_never_read(sample: Path) -> None:
    """Blind overwrites are the exact thing the guard exists to stop."""
    with pytest.raises(PermissionError):
        WriteFileTool()._call(file_path=str(sample), content="clobbered")

    assert sample.read_text() == "alpha\nbeta\ngamma\n"


def test_write_overwrites_after_a_real_read(sample: Path) -> None:
    """Reading first is what unlocks the overwrite."""
    _read(sample)

    result = WriteFileTool()._call(file_path=str(sample), content="replaced\n")

    assert result["action"] == "overwritten"
    assert sample.read_text() == "replaced\n"


def test_write_refuses_when_the_file_changed_since_the_read(sample: Path) -> None:
    """A stale read is as dangerous as no read: someone else edited the file."""
    _read(sample)
    os.utime(sample, (0, 0))

    with pytest.raises(PermissionError):
        WriteFileTool()._call(file_path=str(sample), content="clobbered")


def test_edit_replaces_a_unique_match(sample: Path) -> None:
    """A unique match is edited in place and reported as a diff."""
    _read(sample)

    result = EditFileTool()._call(
        file_path=str(sample), old_string="beta", new_string="delta"
    )

    assert result["replacements"] == 1
    assert result["action"] == "edited"
    assert "-beta" in result["diff"]
    assert "+delta" in result["diff"]
    assert sample.read_text() == "alpha\ndelta\ngamma\n"


def test_edit_refuses_an_ambiguous_match(tmp_path: Path) -> None:
    """Without ``replace_all`` the match must be unique, or nothing is touched."""
    path = tmp_path / "dup.txt"
    path.write_text("x\nx\n")
    _read(path)

    with pytest.raises(ValueError):
        EditFileTool()._call(file_path=str(path), old_string="x", new_string="y")

    assert path.read_text() == "x\nx\n"


def test_edit_replaces_every_occurrence_when_asked(tmp_path: Path) -> None:
    """``replace_all`` turns the ambiguity into the intent."""
    path = tmp_path / "dup.txt"
    path.write_text("x\nx\nx\n")
    _read(path)

    result = EditFileTool()._call(
        file_path=str(path), old_string="x", new_string="y", replace_all=True
    )

    assert result["replacements"] == 3
    assert path.read_text() == "y\ny\ny\n"


def test_edit_refuses_a_file_it_never_read(sample: Path) -> None:
    """Edit carries the same read-first guard as write."""
    with pytest.raises(PermissionError):
        EditFileTool()._call(
            file_path=str(sample), old_string="beta", new_string="delta"
        )


def test_shell_captures_both_streams_and_the_exit_code() -> None:
    """Stdout, stderr and the status are reported separately."""
    result = ShellCommandTool()._call(
        command="printf out; printf err >&2; exit 3",
    )

    assert result["stdout"] == "out"
    assert result["stderr"] == "err"
    assert result["exit_code"] == 3
    assert result["timed_out"] is False


def test_shell_reports_a_timeout_instead_of_raising() -> None:
    """A timeout is data the model can act on, not an exception."""
    result = ShellCommandTool()._call(command="sleep 5", timeout=1)

    assert result["timed_out"] is True
    assert result["exit_code"] == -1


def test_shell_truncates_oversized_output() -> None:
    """Output past the cap is cut with a marker naming the dropped size."""
    text, truncated = ShellCommandTool._truncate("a" * 30_010)

    assert truncated is True
    assert text.endswith("… [10 chars truncated]")


def test_a_failing_tool_call_becomes_a_tool_error_message() -> None:
    """``BaseTool.call`` never raises: the model gets the failure as a result."""
    message = ReadFileTool().call(call_id="c1", file_path="relative.txt")

    assert isinstance(message, ToolErrorMessage)
    assert message.id == "c1"
    assert message.content.startswith("Error while executing ReadFileTool:")


class _FakeDDGS:
    """Stand-in for ``ddgs.DDGS`` returning canned hits.

    Attributes:
        hits: the raw result dicts ``text`` should return.
    """

    hits: List[dict] = []

    def __init__(self, **kwargs: Any) -> None:
        """Accept and ignore the real client's keyword arguments.

        Args:
            **kwargs: the timeout the tool passes.
        """

    def __enter__(self) -> "_FakeDDGS":
        """Enter the context manager the tool opens.

        Returns:
            This same object.
        """
        return self

    def __exit__(self, *exc: Any) -> None:
        """Leave the context manager.

        Args:
            *exc: the exception triple, unused.
        """

    def text(self, query: str, max_results: int) -> List[dict]:
        """Return the canned hits.

        Args:
            query: the search query, unused.
            max_results: the clamped result cap, unused.

        Returns:
            The canned raw hits.
        """
        return self.hits


def test_search_normalizes_every_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The backend's field names are mapped to the tool's own schema."""
    _FakeDDGS.hits = [{"href": "https://a", "title": "A", "body": "about a"}]
    monkeypatch.setattr(search_module, "DDGS", _FakeDDGS)

    result = SearchWebTool()._call(query="a")

    assert result["results"] == [
        {"url": "https://a", "title": "A", "excerpt": "about a"}
    ]
    assert result["timed_out"] is False


def test_search_reports_a_timeout_as_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """A search timeout comes back as an empty, flagged result."""

    class _TimingOutDDGS(_FakeDDGS):
        """A search client that always times out."""

        def text(self, query: str, max_results: int) -> List[dict]:
            """Fail the way ddgs does when nothing arrived in time.

            Args:
                query: the search query, unused.
                max_results: the clamped result cap, unused.

            Raises:
                TimeoutException: always.
            """
            raise TimeoutException("too slow")

    monkeypatch.setattr(search_module, "DDGS", _TimingOutDDGS)

    result = SearchWebTool()._call(query="a")

    assert result == {"query": "a", "results": [], "timed_out": True}


def _stub_trafilatura(monkeypatch: pytest.MonkeyPatch, content: str) -> None:
    """Replace the network layer of the fetch tool with canned page content.

    Args:
        monkeypatch: pytest's patching fixture.
        content: the extracted page text the tool should see.
    """
    monkeypatch.setattr(
        fetch_module,
        "trafilatura",
        SimpleNamespace(
            fetch_url=lambda url, config: "<html/>",
            extract=lambda downloaded, output_format, config: content,
            extract_metadata=lambda downloaded, default_url: SimpleNamespace(
                title="A Page", url=default_url
            ),
        ),
    )
    monkeypatch.setattr(
        fetch_module, "use_config", lambda: SimpleNamespace(set=lambda *args: None)
    )


def test_fetch_answers_the_query_through_the_summary_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The page is downloaded, capped and handed to the injected client."""
    _stub_trafilatura(monkeypatch, "page body")
    client = ScriptedClient([[AssistantMessage(content="the answer")]])

    result = FetchWebTool(client=client)._call(url="https://x", query="what?")

    assert result["answer"] == "the answer"
    assert result["title"] == "A Page"
    assert result["truncated"] is False
    assert "page body" in client.requests[0].message_list[0].content


def test_fetch_joins_only_finalized_assistant_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A streamed fragment is not appended a second time to the answer.

    ``AssistantChunkMessage`` passes an ``isinstance`` check for ``AssistantMessage``,
    so the summariser filters chunks out explicitly.
    """
    _stub_trafilatura(monkeypatch, "page body")
    client = ScriptedClient(
        [[AssistantChunkMessage(content="frag"), AssistantMessage(content="whole")]]
    )

    result = FetchWebTool(client=client)._call(url="https://x", query="what?")

    assert result["answer"] == "whole"


def test_fetch_rejects_a_non_http_url() -> None:
    """Only ``http(s)`` URLs are fetchable."""
    client = ScriptedClient()

    with pytest.raises(ValueError):
        FetchWebTool(client=client)._call(url="file:///etc/passwd", query="what?")
