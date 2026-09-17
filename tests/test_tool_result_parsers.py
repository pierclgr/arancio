"""Tests for how each tool's structured output is rendered for the user.

The parsers own two decisions: what the user reads, and whether a call that raised
nothing is nonetheless a failure. The second one matters most — a shell command exiting
non-zero returns normally, and only the parser turns it into a
:class:`ToolErrorMessage`.
"""

import pytest

from arancio.core.messages import ToolErrorMessage, ToolResultMessage
from arancio.core.parsers.tool_result.base import BaseToolResultParser
from arancio.core.parsers.tool_result.commands.shell import ShellCommandToolResultParser
from arancio.core.parsers.tool_result.files.edit import EditFileToolResultParser
from arancio.core.parsers.tool_result.files.read import ReadFileToolResultParser
from arancio.core.parsers.tool_result.files.write import WriteFileToolResultParser
from arancio.core.parsers.tool_result.web.fetch import FetchWebToolResultParser
from arancio.core.parsers.tool_result.web.search import SearchWebToolResultParser


def test_read_result_ends_with_the_line_range() -> None:
    """The footer tells the model which slice of the file it is looking at."""
    rendered = ReadFileToolResultParser._render(
        {
            "file_path": "/tmp/a.txt",
            "content": "     1\talpha",
            "start_line": 1,
            "end_line": 1,
            "total_lines": 9,
            "truncated_lines": 0,
        }
    )

    assert rendered == "/tmp/a.txt\n     1\talpha\n[lines 1-1 of 9]"


@pytest.mark.parametrize(
    ("output", "expected_footer"),
    [
        ({"total_lines": 0}, "[empty file]"),
        ({"total_lines": 9, "start_line": 0}, "[no lines returned, file has 9 lines]"),
    ],
    ids=["empty-file", "slice-past-the-end"],
)
def test_read_result_explains_an_absent_slice(
    output: dict, expected_footer: str
) -> None:
    """An empty read says why it is empty rather than rendering nothing."""
    assert ReadFileToolResultParser._render(output) == expected_footer


def test_read_result_counts_capped_lines() -> None:
    """Long lines that were cut are reported, so the model knows text is missing."""
    rendered = ReadFileToolResultParser._render(
        {
            "content": "x",
            "start_line": 1,
            "end_line": 1,
            "total_lines": 1,
            "truncated_lines": 2,
        }
    )

    assert rendered.endswith("[2 long lines truncated]")


@pytest.mark.parametrize(
    ("action", "verb"),
    [("created", "Created"), ("overwritten", "Overwrote"), ("something", "Wrote")],
)
def test_write_result_names_what_happened(action: str, verb: str) -> None:
    """The verb distinguishes a new file from a clobbered one."""
    rendered = WriteFileToolResultParser._render(
        {
            "file_path": "/tmp/a.txt",
            "bytes_written": 6,
            "total_lines": 1,
            "action": action,
        }
    )

    assert rendered == f"{verb} /tmp/a.txt (6 bytes, 1 lines)"


def test_edit_result_pluralizes_and_appends_the_diff() -> None:
    """The summary line is followed by the diff the user needs to review."""
    rendered = EditFileToolResultParser._render(
        {
            "file_path": "/tmp/a.txt",
            "replacements": 2,
            "bytes_before": 10,
            "bytes_after": 12,
            "diff": "-old\n+new",
        }
    )

    assert rendered == "Edited /tmp/a.txt (2 replacements, 10 → 12 bytes)\n-old\n+new"


def test_edit_result_uses_the_singular_for_one_replacement() -> None:
    """One replacement reads as one, not as ``1 replacements``."""
    rendered = EditFileToolResultParser._render(
        {
            "file_path": "/tmp/a.txt",
            "replacements": 1,
            "bytes_before": 1,
            "bytes_after": 1,
        }
    )

    assert "1 replacement," in rendered


def test_shell_result_stacks_output_then_status() -> None:
    """Both streams are shown, then the annotations that explain the outcome."""
    rendered = ShellCommandToolResultParser._render(
        {
            "stdout": "out\n",
            "stderr": "err\n",
            "exit_code": 3,
            "timed_out": False,
            "truncated": True,
        }
    )

    assert rendered == "out\nerr\n[exit code 3]\n[output truncated]"


@pytest.mark.parametrize(
    "output",
    [
        {"stdout": "", "stderr": "", "exit_code": 1, "timed_out": False},
        {"stdout": "", "stderr": "", "exit_code": -1, "timed_out": True},
    ],
    ids=["non-zero-exit", "timed-out"],
)
def test_a_shell_command_that_ran_can_still_be_an_error(output: dict) -> None:
    """The tool raised nothing, so only the parser can mark this a failure."""
    message = ShellCommandToolResultParser.parse(call_id="c1", output=output)

    assert isinstance(message, ToolErrorMessage)


def test_a_clean_shell_command_is_a_plain_result() -> None:
    """Exit code zero stays a result, not an error."""
    message = ShellCommandToolResultParser.parse(
        call_id="c1", output={"stdout": "ok", "stderr": "", "exit_code": 0}
    )

    assert type(message) is ToolResultMessage


def test_search_result_numbers_each_hit_and_counts_them() -> None:
    """Every hit gets a number, and the footer repeats the query."""
    rendered = SearchWebToolResultParser._render(
        {
            "query": "python",
            "results": [{"title": "T", "url": "https://a", "excerpt": "E"}],
            "timed_out": False,
        }
    )

    assert rendered == '1. T\n   https://a\n   E\n\n[1 result for "python"]'


def test_search_result_says_so_when_there_is_nothing() -> None:
    """An empty result list is stated, not rendered as blank."""
    assert SearchWebToolResultParser._render({"results": []}) == "[no results]"


def test_a_timed_out_search_is_an_error_with_a_marker() -> None:
    """A timeout both annotates the text and fails the call."""
    message = SearchWebToolResultParser.parse(
        call_id="c1", output={"query": "q", "results": [], "timed_out": True}
    )

    assert isinstance(message, ToolErrorMessage)
    assert message.display_text == "[no results]\n[timed out]"


def test_fetch_result_footers_the_source() -> None:
    """The answer is followed by where it came from."""
    rendered = FetchWebToolResultParser._render(
        {"answer": "42", "title": "T", "url": "https://a", "truncated": True}
    )

    assert rendered == "42\n\n[T — https://a]\n[content truncated]"


def test_fetch_result_states_an_empty_answer() -> None:
    """A blank answer is labelled rather than shown as nothing."""
    assert FetchWebToolResultParser._render({"answer": ""}) == "[no answer]"


def test_non_dict_output_is_passed_through_unrendered() -> None:
    """An error string from a tool becomes the content and the display text."""
    message = BaseToolResultParser.parse(
        call_id="c1", output="Error while executing X: boom", is_error=True
    )

    assert isinstance(message, ToolErrorMessage)
    assert message.content == "Error while executing X: boom"
    assert message.display_text == message.content
