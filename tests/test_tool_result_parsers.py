"""Tests for tool result parsers."""

from arancio.core.messages import ToolErrorMessage, ToolResultMessage
from arancio.core.parsers.tool_result.commands.shell import ShellCommandToolResultParser
from arancio.core.parsers.tool_result.files.edit import EditFileToolResultParser
from arancio.core.parsers.tool_result.files.read import ReadFileToolResultParser
from arancio.core.parsers.tool_result.files.write import WriteFileToolResultParser
from arancio.core.parsers.tool_result.web.fetch import FetchWebToolResultParser
from arancio.core.parsers.tool_result.web.search import SearchWebToolResultParser


def test_shell_command_parser_keeps_stdout_successful() -> None:
    """Stdout-only shell output is a successful tool result."""
    output = {
        "stdout": "ok\n",
        "stderr": "",
        "exit_code": 0,
        "timed_out": False,
        "truncated": False,
    }

    result = ShellCommandToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text="ok",
    )


def test_shell_command_parser_reports_stderr_without_failing() -> None:
    """Stderr is displayed but does not fail the result when exit code is zero."""
    output = {
        "stdout": "",
        "stderr": "warning\n",
        "exit_code": 0,
        "timed_out": False,
        "truncated": False,
    }

    result = ShellCommandToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text="warning",
    )


def test_shell_command_parser_marks_non_zero_exit_code_as_error() -> None:
    """Non-zero shell exit codes fail the tool result."""
    output = {
        "stdout": "",
        "stderr": "fatal\n",
        "exit_code": 2,
        "timed_out": False,
        "truncated": False,
    }

    result = ShellCommandToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        display_text="fatal\n[exit code 2]",
    )


def test_shell_command_parser_marks_timeout_as_error() -> None:
    """Timed-out shell commands fail the tool result."""
    output = {
        "stdout": "partial\n",
        "stderr": "",
        "exit_code": -1,
        "timed_out": True,
        "truncated": False,
    }

    result = ShellCommandToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        display_text="partial\n[timed out]\n[exit code -1]",
    )


def test_shell_command_parser_honors_explicit_error_flag() -> None:
    """Explicit tool execution errors are preserved by the shell parser."""
    output = "Error while executing ShellCommandTool: boom"

    result = ShellCommandToolResultParser.parse(
        call_id="call_1",
        output=output,
        is_error=True,
    )

    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
    )


def test_read_tool_parser_formats_successful_slice() -> None:
    """Read tool output is rendered with the file path and a slice footer."""
    output = {
        "file_path": "/abs/file.txt",
        "content": "     1\thello\n     2\tworld",
        "start_line": 1,
        "end_line": 2,
        "total_lines": 2,
        "truncated_lines": 0,
    }

    result = ReadFileToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text="/abs/file.txt\n     1\thello\n     2\tworld\n[lines 1-2 of 2]",
    )


def test_read_tool_parser_marks_empty_file() -> None:
    """Empty files are surfaced with the file path and an explicit marker."""
    output = {
        "file_path": "/abs/file.txt",
        "content": "",
        "start_line": 0,
        "end_line": 0,
        "total_lines": 0,
        "truncated_lines": 0,
    }

    result = ReadFileToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text="/abs/file.txt\n[empty file]",
    )


def test_read_tool_parser_marks_offset_past_end() -> None:
    """Offsets past the last line yield the file path and a descriptive marker."""
    output = {
        "file_path": "/abs/file.txt",
        "content": "",
        "start_line": 0,
        "end_line": 0,
        "total_lines": 5,
        "truncated_lines": 0,
    }

    result = ReadFileToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text="/abs/file.txt\n[no lines returned, file has 5 lines]",
    )


def test_read_tool_parser_reports_truncated_lines() -> None:
    """Truncated lines are reported in a trailing footer line."""
    output = {
        "file_path": "/abs/file.txt",
        "content": "     1\thello… [line truncated]",
        "start_line": 1,
        "end_line": 1,
        "total_lines": 1,
        "truncated_lines": 1,
    }

    result = ReadFileToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text=(
            "/abs/file.txt\n     1\thello… [line truncated]\n[lines 1-1 of 1]\n"
            "[1 long lines truncated]"
        ),
    )


def test_read_tool_parser_honors_explicit_error_flag() -> None:
    """Explicit tool execution errors are preserved by the read parser."""
    output = "Error while executing ReadFileTool: boom"

    result = ReadFileToolResultParser.parse(
        call_id="call_1",
        output=output,
        is_error=True,
    )

    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
    )


def test_write_tool_parser_reports_created() -> None:
    """Created files are rendered with the ``Created`` verb."""
    output = {
        "file_path": "/abs/new.txt",
        "bytes_written": 12,
        "action": "created",
        "total_lines": 2,
    }

    result = WriteFileToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text="Created /abs/new.txt (12 bytes, 2 lines)",
    )


def test_write_tool_parser_reports_overwritten() -> None:
    """Overwritten files are rendered with the ``Overwrote`` verb."""
    output = {
        "file_path": "/abs/file.txt",
        "bytes_written": 5,
        "action": "overwritten",
        "total_lines": 1,
    }

    result = WriteFileToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text="Overwrote /abs/file.txt (5 bytes, 1 lines)",
    )


def test_write_tool_parser_honors_explicit_error_flag() -> None:
    """Explicit tool execution errors are preserved by the write parser."""
    output = "Error while executing WriteFileTool: boom"

    result = WriteFileToolResultParser.parse(
        call_id="call_1",
        output=output,
        is_error=True,
    )

    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
    )


def test_edit_tool_parser_reports_single_replacement() -> None:
    """A single replacement is rendered with the singular noun."""
    output = {
        "file_path": "/abs/file.py",
        "replacements": 1,
        "bytes_before": 12,
        "bytes_after": 13,
        "action": "edited",
    }

    result = EditFileToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text="Edited /abs/file.py (1 replacement, 12 → 13 bytes)",
    )


def test_edit_tool_parser_reports_plural_replacements() -> None:
    """Multiple replacements are rendered with the plural noun."""
    output = {
        "file_path": "/abs/file.py",
        "replacements": 3,
        "bytes_before": 30,
        "bytes_after": 33,
        "action": "edited",
    }

    result = EditFileToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text="Edited /abs/file.py (3 replacements, 30 → 33 bytes)",
    )


def test_edit_tool_parser_appends_diff() -> None:
    """A unified diff in the output is appended below the summary line."""
    diff = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-x = 1\n+x = 2"
    output = {
        "file_path": "/abs/file.py",
        "replacements": 1,
        "bytes_before": 5,
        "bytes_after": 5,
        "action": "edited",
        "diff": diff,
    }

    result = EditFileToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text=f"Edited /abs/file.py (1 replacement, 5 → 5 bytes)\n{diff}",
    )


def test_edit_tool_parser_honors_explicit_error_flag() -> None:
    """Explicit tool execution errors are preserved by the edit parser."""
    output = "Error while executing EditFileTool: boom"

    result = EditFileToolResultParser.parse(
        call_id="call_1",
        output=output,
        is_error=True,
    )

    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
    )


# — SearchWebToolResultParser ——————————————————————————————————————————


def test_search_web_parser_formats_results_with_footer() -> None:
    """Successful results render as a numbered list plus the query footer."""
    output = {
        "query": "pytorch",
        "results": [
            {
                "url": "https://pytorch.org/docs/",
                "title": "PyTorch docs",
                "excerpt": "Docs excerpt.",
            },
            {
                "url": "https://pytorch.org/tutorials/",
                "title": "PyTorch tutorials",
                "excerpt": "Tut excerpt.",
            },
        ],
        "timed_out": False,
    }

    result = SearchWebToolResultParser.parse(call_id="call_1", output=output)

    expected_content = (
        "1. PyTorch docs\n"
        "   https://pytorch.org/docs/\n"
        "   Docs excerpt.\n"
        "\n"
        "2. PyTorch tutorials\n"
        "   https://pytorch.org/tutorials/\n"
        "   Tut excerpt.\n"
        "\n"
        '[2 results for "pytorch"]'
    )
    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text=expected_content,
    )


def test_search_web_parser_singular_noun() -> None:
    """A single result uses the singular ``result`` noun in the footer."""
    output = {
        "query": "only one",
        "results": [
            {"url": "https://x", "title": "T", "excerpt": "E"},
        ],
        "timed_out": False,
    }

    result = SearchWebToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text=('1. T\n   https://x\n   E\n\n[1 result for "only one"]'),
    )


def test_search_web_parser_empty_results() -> None:
    """Empty result list renders the ``[no results]`` marker."""
    output = {"query": "no hits", "results": [], "timed_out": False}

    result = SearchWebToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text="[no results]",
    )


def test_search_web_parser_timeout() -> None:
    """Timeout dicts are surfaced as tool errors with the ``[timed out]`` tag."""
    output = {"query": "slow", "results": [], "timed_out": True}

    result = SearchWebToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        display_text="[no results]\n[timed out]",
    )


def test_search_web_parser_explicit_error_flag() -> None:
    """Explicit tool execution errors are preserved by the web search parser."""
    output = "Error while executing SearchWebTool: rate limited"

    result = SearchWebToolResultParser.parse(
        call_id="call_1",
        output=output,
        is_error=True,
    )

    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
    )


# — FetchWebToolResultParser ———————————————————————————————————————————


def test_fetch_web_parser_formats_answer_with_footer() -> None:
    """Successful output renders the answer plus a title/url footer."""
    output = {
        "url": "https://x/final",
        "query": "what does it cover?",
        "title": "Title",
        "content_type": None,
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "answer": "The page covers X.",
        "truncated": False,
    }

    result = FetchWebToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text="The page covers X.\n\n[Title — https://x/final]",
    )


def test_fetch_web_parser_appends_truncated_marker() -> None:
    """Truncated content adds a ``[content truncated]`` marker to the footer."""
    output = {
        "url": "https://x",
        "query": "q",
        "title": "T",
        "content_type": None,
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "answer": "answer",
        "truncated": True,
    }

    result = FetchWebToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text="answer\n\n[T — https://x]\n[content truncated]",
    )


def test_fetch_web_parser_empty_answer_marker() -> None:
    """An empty answer renders the ``[no answer]`` marker."""
    output = {
        "url": "https://x",
        "query": "q",
        "title": None,
        "content_type": None,
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "answer": "",
        "truncated": False,
    }

    result = FetchWebToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text="[no answer]\n\n[https://x]",
    )


def test_fetch_web_parser_explicit_error_flag() -> None:
    """Explicit tool execution errors are preserved by the web fetch parser."""
    output = "Error while executing FetchWebTool: failed to fetch https://x"

    result = FetchWebToolResultParser.parse(
        call_id="call_1",
        output=output,
        is_error=True,
    )

    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
    )
