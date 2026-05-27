"""Tests for tool result parsers."""

from codo.parsers.tool_result.commands.shell import ShellCommandToolResultParser
from codo.parsers.tool_result.files.edit import EditFileToolResultParser
from codo.parsers.tool_result.files.glob import GlobToolResultParser
from codo.parsers.tool_result.files.grep import GrepToolResultParser
from codo.parsers.tool_result.files.read import ReadFileToolResultParser
from codo.parsers.tool_result.files.write import WriteFileToolResultParser
from codo.parsers.tool_result.web.search import SearchWebToolResultParser
from codo.types.messages import ToolErrorMessage, ToolResultMessage


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
        content="ok",
        id="call_1",
        output=output,
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
        content="warning",
        id="call_1",
        output=output,
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
        content="fatal\n[exit code 2]",
        id="call_1",
        output=output,
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
        content="partial\n[timed out]\n[exit code -1]",
        id="call_1",
        output=output,
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
        output=output,
    )


def test_read_tool_parser_formats_successful_slice() -> None:
    """Read tool output is rendered with a footer summarizing the slice."""
    output = {
        "content": "     1\thello\n     2\tworld",
        "start_line": 1,
        "end_line": 2,
        "total_lines": 2,
        "truncated_lines": 0,
    }

    result = ReadFileToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content="     1\thello\n     2\tworld\n[lines 1-2 of 2]",
        id="call_1",
        output=output,
    )


def test_read_tool_parser_marks_empty_file() -> None:
    """Empty files are surfaced with an explicit marker."""
    output = {
        "content": "",
        "start_line": 0,
        "end_line": 0,
        "total_lines": 0,
        "truncated_lines": 0,
    }

    result = ReadFileToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content="[empty file]",
        id="call_1",
        output=output,
    )


def test_read_tool_parser_marks_offset_past_end() -> None:
    """Offsets past the last line yield a descriptive marker."""
    output = {
        "content": "",
        "start_line": 0,
        "end_line": 0,
        "total_lines": 5,
        "truncated_lines": 0,
    }

    result = ReadFileToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content="[no lines returned, file has 5 lines]",
        id="call_1",
        output=output,
    )


def test_read_tool_parser_reports_truncated_lines() -> None:
    """Truncated lines are reported in a trailing footer line."""
    output = {
        "content": "     1\thello… [line truncated]",
        "start_line": 1,
        "end_line": 1,
        "total_lines": 1,
        "truncated_lines": 1,
    }

    result = ReadFileToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content=(
            "     1\thello… [line truncated]\n[lines 1-1 of 1]\n"
            "[1 long lines truncated]"
        ),
        id="call_1",
        output=output,
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
        output=output,
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
        content="Created /abs/new.txt (12 bytes, 2 lines)",
        id="call_1",
        output=output,
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
        content="Overwrote /abs/file.txt (5 bytes, 1 lines)",
        id="call_1",
        output=output,
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
        output=output,
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
        content="Edited /abs/file.py (1 replacement, 12 → 13 bytes)",
        id="call_1",
        output=output,
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
        content="Edited /abs/file.py (3 replacements, 30 → 33 bytes)",
        id="call_1",
        output=output,
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
        output=output,
    )


# — GrepToolResultParser ——————————————————————————————————————————————


def test_grep_parser_files_with_matches() -> None:
    """Files-with-matches output is formatted with one path per line and a footer."""
    output = {
        "matches": ["/abs/a.py", "/abs/b.py"],
        "total_matches": 2,
        "truncated": False,
        "timed_out": False,
        "exit_code": 0,
        "output_mode": "files_with_matches",
    }

    result = GrepToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content="/abs/a.py\n/abs/b.py\n[2 files matched]",
        id="call_1",
        output=output,
    )


def test_grep_parser_content_mode() -> None:
    """Content mode formats entries as file:line:text with a summary footer."""
    output = {
        "matches": [
            {
                "file": "/abs/a.py",
                "line": 3,
                "content": "def foo():",
                "is_context": False,
            },
            {
                "file": "/abs/b.py",
                "line": 7,
                "content": "def bar():",
                "is_context": False,
            },
        ],
        "total_matches": 2,
        "truncated": False,
        "timed_out": False,
        "exit_code": 0,
        "output_mode": "content",
    }

    result = GrepToolResultParser.parse(call_id="call_1", output=output)

    expected = (
        "/abs/a.py:3:def foo():\n/abs/b.py:7:def bar():\n[2 matches across 2 files]"
    )
    assert result == ToolResultMessage(
        content=expected,
        id="call_1",
        output=output,
    )


def test_grep_parser_count_mode() -> None:
    """Count mode formats entries as file:N with a summary footer."""
    output = {
        "matches": [
            {"file": "/abs/a.py", "count": 5},
            {"file": "/abs/b.py", "count": 3},
        ],
        "total_matches": 2,
        "truncated": False,
        "timed_out": False,
        "exit_code": 0,
        "output_mode": "count",
    }

    result = GrepToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content="/abs/a.py:5\n/abs/b.py:3\n[2 files with matches]",
        id="call_1",
        output=output,
    )


def test_grep_parser_truncated_output() -> None:
    """Truncated output includes a different footer indicating the cap."""
    output = {
        "matches": ["/abs/a.py"],
        "total_matches": 42,
        "truncated": True,
        "timed_out": False,
        "exit_code": 0,
        "output_mode": "files_with_matches",
    }

    result = GrepToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content="/abs/a.py\n[output truncated, showing first 1 of 42]",
        id="call_1",
        output=output,
    )


def test_grep_parser_timeout() -> None:
    """Timed-out results are surfaced as tool errors."""
    output = {
        "matches": [],
        "total_matches": 0,
        "truncated": False,
        "timed_out": True,
        "exit_code": -1,
        "output_mode": "content",
    }

    result = GrepToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolErrorMessage(
        content="[0 matches across 0 files]\n[timed out]",
        id="call_1",
        output=output,
    )


def test_grep_parser_explicit_error_flag() -> None:
    """Explicit tool execution errors are preserved by the grep parser."""
    output = "Error while executing GrepTool: boom"

    result = GrepToolResultParser.parse(
        call_id="call_1",
        output=output,
        is_error=True,
    )

    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        output=output,
    )


# — GlobToolResultParser ——————————————————————————————————————————————


def test_glob_parser_formats_matches_with_footer() -> None:
    """Successful glob output is rendered as paths plus an mtime footer."""
    output = {
        "matches": ["/abs/b.py", "/abs/a.py"],
        "total_matches": 2,
        "truncated": False,
        "timed_out": False,
        "exit_code": 0,
        "search_path": "/abs",
        "pattern": "*.py",
    }

    result = GlobToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content="/abs/b.py\n/abs/a.py\n[2 files matched, sorted by mtime]",
        id="call_1",
        output=output,
    )


def test_glob_parser_empty_match() -> None:
    """Empty match dicts render as a single ``[no files matched]`` marker."""
    output = {
        "matches": [],
        "total_matches": 0,
        "truncated": False,
        "timed_out": False,
        "exit_code": 1,
        "search_path": "/abs",
        "pattern": "*.nope",
    }

    result = GlobToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content="[no files matched]",
        id="call_1",
        output=output,
    )


def test_glob_parser_singular_noun() -> None:
    """A single match uses the singular ``file`` noun in the footer."""
    output = {
        "matches": ["/abs/only.py"],
        "total_matches": 1,
        "truncated": False,
        "timed_out": False,
        "exit_code": 0,
        "search_path": "/abs",
        "pattern": "*.py",
    }

    result = GlobToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content="/abs/only.py\n[1 file matched, sorted by mtime]",
        id="call_1",
        output=output,
    )


def test_glob_parser_truncated_output() -> None:
    """Truncated output uses the truncation footer with the original total."""
    output = {
        "matches": ["/abs/a.py"],
        "total_matches": 42,
        "truncated": True,
        "timed_out": False,
        "exit_code": 0,
        "search_path": "/abs",
        "pattern": "*.py",
    }

    result = GlobToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content="/abs/a.py\n[output truncated, showing first 1 of 42]",
        id="call_1",
        output=output,
    )


def test_glob_parser_timeout() -> None:
    """Timed-out results are surfaced as tool errors with the ``[timed out]`` tag."""
    output = {
        "matches": [],
        "total_matches": 0,
        "truncated": False,
        "timed_out": True,
        "exit_code": -1,
        "search_path": "/abs",
        "pattern": "*.py",
    }

    result = GlobToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolErrorMessage(
        content="[no files matched]\n[timed out]",
        id="call_1",
        output=output,
    )


def test_glob_parser_explicit_error_flag() -> None:
    """Explicit tool execution errors are preserved by the glob parser."""
    output = "Error while executing GlobTool: boom"

    result = GlobToolResultParser.parse(
        call_id="call_1",
        output=output,
        is_error=True,
    )

    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        output=output,
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
        content=expected_content,
        id="call_1",
        output=output,
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
        content=('1. T\n   https://x\n   E\n\n[1 result for "only one"]'),
        id="call_1",
        output=output,
    )


def test_search_web_parser_empty_results() -> None:
    """Empty result list renders the ``[no results]`` marker."""
    output = {"query": "no hits", "results": [], "timed_out": False}

    result = SearchWebToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolResultMessage(
        content="[no results]",
        id="call_1",
        output=output,
    )


def test_search_web_parser_timeout() -> None:
    """Timeout dicts are surfaced as tool errors with the ``[timed out]`` tag."""
    output = {"query": "slow", "results": [], "timed_out": True}

    result = SearchWebToolResultParser.parse(call_id="call_1", output=output)

    assert result == ToolErrorMessage(
        content="[no results]\n[timed out]",
        id="call_1",
        output=output,
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
        output=output,
    )
