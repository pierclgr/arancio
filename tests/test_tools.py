"""Tests for tool execution helpers."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, List
from unittest.mock import MagicMock, patch

import pytest
import yaml
from dynamic_markdown.types.files.base import DynamicMarkdownFile

from arancio.core.messages import (
    AssistantMessage,
    ToolErrorMessage,
    ToolResultMessage,
)
from arancio.core.parsers.tool_result.base import BaseToolResultParser
from arancio.core.tools.base import BaseTool
from arancio.core.tools.commands.shell import ShellCommandTool
from arancio.core.tools.files.edit import EditFileTool
from arancio.core.tools.files.read import ReadFileTool
from arancio.core.tools.files.write import WriteFileTool
from arancio.core.tools.schema import ToolSchema
from arancio.core.tools.session import default_session
from arancio.core.tools.web.fetch import FetchWebTool
from arancio.core.tools.web.search import SearchWebTool


@pytest.fixture(autouse=True)
def _reset_default_tool_session() -> None:
    """Clear the shared tool session before each test for isolation."""
    default_session.clear()


class _RecordingToolResultParser(BaseToolResultParser):
    """Record parser inputs and return a tool result message."""

    calls: ClassVar[List[dict[str, Any]]] = []

    @classmethod
    def parse(
        cls,
        call_id: str,
        output: Any,
        is_error: bool = False,
    ) -> ToolResultMessage:
        """Record parser inputs.

        Args:
            call_id: identifier of the tool call this result answers.
            output: raw tool output.
            is_error: whether the tool execution failed.

        Returns:
            A parsed tool result message.
        """
        cls.calls.append(
            {
                "call_id": call_id,
                "output": output,
                "is_error": is_error,
            }
        )
        result_class = ToolErrorMessage if is_error else ToolResultMessage
        return result_class(
            content=f"parsed: {output}",
            id=call_id,
        )


class FailingTool(BaseTool):
    """Tool raising during raw execution."""

    _result_parser = _RecordingToolResultParser

    def __init__(self) -> None:
        """Set description and input schema directly, skipping disk load."""
        self.description = "Failing tool."
        self.input_schema = {"type": "object", "properties": {}}

    def _call(self, **kwargs) -> str:
        """Raise during raw execution.

        Args:
            **kwargs: ignored tool arguments.

        Raises:
            RuntimeError: always raised for this test helper.
        """
        raise RuntimeError("boom")


def test_tool_exception_routes_through_result_parser() -> None:
    """Tool exceptions are converted by the configured result parser."""
    _RecordingToolResultParser.calls = []

    result = FailingTool().call(call_id="call_1")

    output = "Error while executing FailingTool: boom"
    assert _RecordingToolResultParser.calls == [
        {
            "call_id": "call_1",
            "output": output,
            "is_error": True,
        }
    ]
    assert result == ToolErrorMessage(
        content=f"parsed: {output}",
        id="call_1",
    )


@patch("arancio.core.tools.commands.shell.subprocess.run")
@patch("arancio.core.tools.commands.shell.shutil.which")
@patch("arancio.core.tools.commands.shell.platform.system")
def test_shell_command_windows_prefers_windows_powershell(
    system_mock,
    which_mock,
    run_mock,
) -> None:
    """On Windows, the tool prefers powershell.exe when it is available."""
    system_mock.return_value = "Windows"
    which_mock.side_effect = lambda name: (
        "powershell.exe" if name == "powershell.exe" else None
    )
    run_mock.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="ok\n",
        stderr="",
    )

    result = ShellCommandTool().call(
        call_id="call_1",
        command="Write-Output ok",
        timeout=5,
        cwd="C:\\repo",
    )

    run_mock.assert_called_once_with(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Write-Output ok",
        ],
        shell=False,
        cwd="C:\\repo",
        capture_output=True,
        text=True,
        timeout=5,
    )
    output = {
        "stdout": "ok\n",
        "stderr": "",
        "exit_code": 0,
        "timed_out": False,
        "truncated": False,
    }
    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text="ok",
    )


@patch("arancio.core.tools.commands.shell.subprocess.run")
@patch("arancio.core.tools.commands.shell.shutil.which")
@patch("arancio.core.tools.commands.shell.platform.system")
def test_shell_command_windows_falls_back_to_pwsh(
    system_mock, which_mock, run_mock
) -> None:
    """On Windows, the tool falls back to pwsh when powershell.exe is unavailable."""
    system_mock.return_value = "Windows"
    which_mock.side_effect = lambda name: "pwsh" if name == "pwsh" else None
    run_mock.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="ok\n",
        stderr="",
    )

    ShellCommandTool().call(call_id="call_1", command="Write-Output ok")

    command_args = run_mock.call_args.args[0]
    assert command_args[0] == "pwsh"
    assert command_args[-1] == "Write-Output ok"


@patch("arancio.core.tools.commands.shell.subprocess.run")
@patch("arancio.core.tools.commands.shell.shutil.which")
@patch("arancio.core.tools.commands.shell.platform.system")
def test_shell_command_windows_reports_missing_host(
    system_mock, which_mock, run_mock
) -> None:
    """Missing PowerShell hosts are reported as tool errors."""
    system_mock.return_value = "Windows"
    which_mock.return_value = None

    result = ShellCommandTool().call(call_id="call_1", command="Get-ChildItem")

    run_mock.assert_not_called()
    output = (
        "Error while executing ShellCommandTool: "
        "PowerShell host not found: powershell.exe or pwsh"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
    )


@patch("arancio.core.tools.commands.shell.subprocess.run")
@patch("arancio.core.tools.commands.shell.shutil.which")
@patch("arancio.core.tools.commands.shell.platform.system")
def test_shell_command_windows_reports_timeout(
    system_mock, which_mock, run_mock
) -> None:
    """Timed-out PowerShell commands return partial output as an error."""
    system_mock.return_value = "Windows"
    which_mock.return_value = "powershell.exe"
    run_mock.side_effect = subprocess.TimeoutExpired(
        cmd=[],
        timeout=1,
        output=b"partial\n",
        stderr=b"slow\n",
    )

    result = ShellCommandTool().call(
        call_id="call_1",
        command="Start-Sleep 10",
        timeout=1,
    )

    output = {
        "stdout": "partial\n",
        "stderr": "slow\n",
        "exit_code": -1,
        "timed_out": True,
        "truncated": False,
    }
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        display_text="partial\nslow\n[timed out]\n[exit code -1]",
    )


@patch("arancio.core.tools.commands.shell.subprocess.run")
@patch("arancio.core.tools.commands.shell.shutil.which")
@patch("arancio.core.tools.commands.shell.platform.system")
def test_shell_command_windows_truncates_long_output(
    system_mock, which_mock, run_mock
) -> None:
    """PowerShell command streams are truncated independently."""
    system_mock.return_value = "Windows"
    which_mock.return_value = "powershell.exe"
    run_mock.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="a" * 30_001,
        stderr="",
    )

    output = ShellCommandTool()._call(command="Write-Output long")

    assert output["stdout"].startswith("a" * 30_000)
    assert "1 chars truncated" in output["stdout"]
    assert output["truncated"] is True


@patch("arancio.core.tools.commands.shell.subprocess.run")
@patch("arancio.core.tools.commands.shell.platform.system")
def test_shell_command_posix_runs_via_sh(system_mock, run_mock) -> None:
    """On macOS/Linux, the tool runs the command via ``/bin/sh -c``."""
    system_mock.return_value = "Darwin"
    run_mock.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="ok\n",
        stderr="",
    )

    result = ShellCommandTool().call(
        call_id="call_1",
        command="echo ok",
        timeout=5,
        cwd="/repo",
    )

    run_mock.assert_called_once_with(
        "echo ok",
        shell=True,
        cwd="/repo",
        capture_output=True,
        text=True,
        timeout=5,
    )
    output = {
        "stdout": "ok\n",
        "stderr": "",
        "exit_code": 0,
        "timed_out": False,
        "truncated": False,
    }
    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text="ok",
    )


@patch("arancio.core.tools.commands.shell.subprocess.run")
@patch("arancio.core.tools.commands.shell.platform.system")
def test_shell_command_posix_reports_timeout(system_mock, run_mock) -> None:
    """Timed-out POSIX shell commands return partial output as an error."""
    system_mock.return_value = "Linux"
    run_mock.side_effect = subprocess.TimeoutExpired(
        cmd="sleep 10",
        timeout=1,
        output=b"partial\n",
        stderr=b"slow\n",
    )

    result = ShellCommandTool().call(
        call_id="call_1",
        command="sleep 10",
        timeout=1,
    )

    output = {
        "stdout": "partial\n",
        "stderr": "slow\n",
        "exit_code": -1,
        "timed_out": True,
        "truncated": False,
    }
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        display_text="partial\nslow\n[timed out]\n[exit code -1]",
    )


@patch("arancio.core.tools.commands.shell.subprocess.run")
@patch("arancio.core.tools.commands.shell.platform.system")
def test_shell_command_posix_truncates_long_output(system_mock, run_mock) -> None:
    """POSIX shell command streams are truncated independently."""
    system_mock.return_value = "Linux"
    run_mock.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="a" * 30_001,
        stderr="",
    )

    output = ShellCommandTool()._call(command="echo long")

    assert output["stdout"].startswith("a" * 30_000)
    assert "1 chars truncated" in output["stdout"]
    assert output["truncated"] is True


def test_read_tool_reads_full_file_with_line_numbers(tmp_path: Path) -> None:
    """ReadFileTool returns each line prefixed with its 1-indexed line number."""
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\ngamma\n")

    result = ReadFileTool().call(call_id="call_1", file_path=str(target))

    canonical = str(target.resolve())
    expected_block = "     1\talpha\n     2\tbeta\n     3\tgamma"
    output = {
        "file_path": canonical,
        "content": expected_block,
        "start_line": 1,
        "end_line": 3,
        "total_lines": 3,
        "truncated_lines": 0,
    }
    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text=f"{canonical}\n{expected_block}\n[lines 1-3 of 3]",
    )


def test_read_tool_slices_with_offset_and_limit(tmp_path: Path) -> None:
    """ReadFileTool respects offset and limit while preserving line numbers."""
    target = tmp_path / "sample.txt"
    target.write_text("\n".join(f"line-{i}" for i in range(1, 6)) + "\n")

    result = ReadFileTool().call(
        call_id="call_1",
        file_path=str(target),
        offset=2,
        limit=2,
    )

    canonical = str(target.resolve())
    expected_block = "     2\tline-2\n     3\tline-3"
    output = {
        "file_path": canonical,
        "content": expected_block,
        "start_line": 2,
        "end_line": 3,
        "total_lines": 5,
        "truncated_lines": 0,
    }
    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text=f"{canonical}\n{expected_block}\n[lines 2-3 of 5]",
    )


def test_read_tool_truncates_long_lines(tmp_path: Path) -> None:
    """Lines exceeding the per-line cap are truncated with a marker."""
    long_line = "a" * (ReadFileTool._max_line_chars + 25)
    target = tmp_path / "sample.txt"
    target.write_text(long_line + "\n")

    result = ReadFileTool().call(call_id="call_1", file_path=str(target))

    canonical = str(target.resolve())
    capped = "a" * ReadFileTool._max_line_chars + ReadFileTool._line_truncation_marker
    expected_block = f"     1\t{capped}"
    output = {
        "file_path": canonical,
        "content": expected_block,
        "start_line": 1,
        "end_line": 1,
        "total_lines": 1,
        "truncated_lines": 1,
    }
    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text=(
            f"{canonical}\n{expected_block}\n[lines 1-1 of 1]\n[1 long lines truncated]"
        ),
    )


def test_read_tool_handles_empty_file(tmp_path: Path) -> None:
    """Empty files return an empty content block and zero counts."""
    target = tmp_path / "empty.txt"
    target.write_text("")

    result = ReadFileTool().call(call_id="call_1", file_path=str(target))

    canonical = str(target.resolve())
    output = {
        "file_path": canonical,
        "content": "",
        "start_line": 0,
        "end_line": 0,
        "total_lines": 0,
        "truncated_lines": 0,
    }
    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text=f"{canonical}\n[empty file]",
    )


def test_read_tool_expands_md_file_via_dynamic_markdown(tmp_path: Path) -> None:
    """A .md file is expanded through dynamic_markdown before line numbering."""
    (tmp_path / "other.md").write_text("included")
    target = tmp_path / "doc.md"
    target.write_text("before <include>other.md</include> after")

    result = ReadFileTool().call(call_id="call_1", file_path=str(target))

    canonical = str(target.resolve())
    expected_block = "     1\tbefore included after"
    output = {
        "file_path": canonical,
        "content": expected_block,
        "start_line": 1,
        "end_line": 1,
        "total_lines": 1,
        "truncated_lines": 0,
    }
    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text=f"{canonical}\n{expected_block}\n[lines 1-1 of 1]",
    )


def test_read_tool_non_md_file_is_not_expanded(tmp_path: Path) -> None:
    """A non-.md file containing dynamic-markdown-like tags is read raw."""
    target = tmp_path / "doc.txt"
    target.write_text("before <include>other.md</include> after")

    result = ReadFileTool().call(call_id="call_1", file_path=str(target))

    assert "<include>other.md</include>" in result.content["content"]


def test_read_tool_md_file_with_unresolvable_field_tag_raises(tmp_path: Path) -> None:
    """A .md file's dynamic-markdown expansion failure surfaces as a tool error."""
    target = tmp_path / "doc.md"
    target.write_text("<field>name</field>")

    result = ReadFileTool().call(call_id="call_1", file_path=str(target))

    assert isinstance(result, ToolErrorMessage)
    assert "Error while executing ReadFileTool" in result.content
    assert "requires a field_source" in result.content


def test_read_tool_rejects_relative_paths() -> None:
    """Relative paths surface as tool errors via the wrapper."""
    result = ReadFileTool().call(call_id="call_1", file_path="relative.txt")

    output = (
        "Error while executing ReadFileTool: file_path must be absolute: relative.txt"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
    )


def test_read_tool_reports_missing_file(tmp_path: Path) -> None:
    """Missing files surface as tool errors via the wrapper."""
    target = tmp_path / "missing.txt"

    result = ReadFileTool().call(call_id="call_1", file_path=str(target))

    output = f"Error while executing ReadFileTool: File not found: {target}"
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
    )


def test_read_tool_reports_directory_path(tmp_path: Path) -> None:
    """Directory paths surface as tool errors via the wrapper."""
    result = ReadFileTool().call(call_id="call_1", file_path=str(tmp_path))

    output = (
        f"Error while executing ReadFileTool: Path is not a regular file: {tmp_path}"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
    )


def test_read_tool_registers_read_with_session(tmp_path: Path) -> None:
    """Successful reads register the file in the shared tool session."""
    target = tmp_path / "sample.txt"
    target.write_text("hello\n")

    ReadFileTool().call(call_id="call_1", file_path=str(target))

    canonical = str(target.resolve())
    assert default_session.is_known(canonical) is True
    assert default_session.is_fresh(canonical, target.stat().st_mtime) is True


def test_write_tool_creates_new_file(tmp_path: Path) -> None:
    """Writing to a path that does not exist creates the file."""
    target = tmp_path / "new.txt"

    result = WriteFileTool().call(
        call_id="call_1",
        file_path=str(target),
        content="hello\nworld\n",
    )

    assert target.read_text() == "hello\nworld\n"
    canonical = str(target.resolve())
    output = {
        "file_path": canonical,
        "bytes_written": 12,
        "action": "created",
        "total_lines": 2,
    }
    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text=f"Created {canonical} (12 bytes, 2 lines)",
    )


def test_write_tool_overwrites_after_read(tmp_path: Path) -> None:
    """Pre-reading the file unlocks an overwrite of an existing target."""
    target = tmp_path / "sample.txt"
    target.write_text("old")

    ReadFileTool().call(call_id="call_1", file_path=str(target))
    result = WriteFileTool().call(
        call_id="call_2",
        file_path=str(target),
        content="new content",
    )

    assert target.read_text() == "new content"
    canonical = str(target.resolve())
    output = {
        "file_path": canonical,
        "bytes_written": 11,
        "action": "overwritten",
        "total_lines": 1,
    }
    assert result == ToolResultMessage(
        content=output,
        id="call_2",
        display_text=f"Overwrote {canonical} (11 bytes, 1 lines)",
    )


def test_write_tool_refuses_overwrite_without_prior_read(tmp_path: Path) -> None:
    """Existing files that were not read this session cannot be overwritten."""
    target = tmp_path / "sample.txt"
    target.write_text("untouched")

    result = WriteFileTool().call(
        call_id="call_1",
        file_path=str(target),
        content="should not land",
    )

    assert target.read_text() == "untouched"
    output = (
        f"Error while executing WriteFileTool: "
        f"File exists but was not read this session; "
        f"read it first before overwriting: {target}"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
    )


def test_write_tool_refuses_overwrite_after_mtime_drift(tmp_path: Path) -> None:
    """Stale reads (mtime drifted since read) are rejected with a clear error."""
    target = tmp_path / "sample.txt"
    target.write_text("old")

    original_mtime = target.stat().st_mtime
    ReadFileTool().call(call_id="call_1", file_path=str(target))

    drifted = original_mtime + 100.0
    os.utime(target, (drifted, drifted))

    result = WriteFileTool().call(
        call_id="call_2",
        file_path=str(target),
        content="should not land",
    )

    assert target.read_text() == "old"
    output = (
        f"Error while executing WriteFileTool: "
        f"File has changed on disk since it was read; "
        f"re-read before overwriting: {target}"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_2",
    )


def test_write_tool_rejects_relative_paths() -> None:
    """Relative paths surface as tool errors via the wrapper."""
    result = WriteFileTool().call(
        call_id="call_1",
        file_path="relative.txt",
        content="x",
    )

    output = (
        "Error while executing WriteFileTool: file_path must be absolute: relative.txt"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
    )


def test_write_tool_creates_missing_parent_dirs(tmp_path: Path) -> None:
    """Missing parent directories are created automatically."""
    target = tmp_path / "missing_dir" / "nested" / "file.txt"

    result = WriteFileTool().call(
        call_id="call_1",
        file_path=str(target),
        content="x",
    )

    assert target.read_text() == "x"
    canonical = str(target.resolve())
    output = {
        "file_path": canonical,
        "bytes_written": 1,
        "action": "created",
        "total_lines": 1,
    }
    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text=f"Created {canonical} (1 bytes, 1 lines)",
    )


def test_write_tool_reports_directory_target(tmp_path: Path) -> None:
    """Directory targets surface as tool errors via the wrapper."""
    result = WriteFileTool().call(
        call_id="call_1",
        file_path=str(tmp_path),
        content="x",
    )

    output = (
        f"Error while executing WriteFileTool: Path is not a regular file: {tmp_path}"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
    )


def test_write_tool_handles_empty_content(tmp_path: Path) -> None:
    """Empty content creates a zero-byte file with zero lines."""
    target = tmp_path / "empty.txt"

    result = WriteFileTool().call(
        call_id="call_1",
        file_path=str(target),
        content="",
    )

    assert target.read_bytes() == b""
    canonical = str(target.resolve())
    output = {
        "file_path": canonical,
        "bytes_written": 0,
        "action": "created",
        "total_lines": 0,
    }
    assert result == ToolResultMessage(
        content=output,
        id="call_1",
        display_text=f"Created {canonical} (0 bytes, 0 lines)",
    )


def test_edit_tool_replaces_unique_match(tmp_path: Path) -> None:
    """A unique substring is replaced and the file content updated on disk."""
    target = tmp_path / "sample.py"
    target.write_text("x = 1\ny = 2\n")

    ReadFileTool().call(call_id="call_1", file_path=str(target))
    result = EditFileTool().call(
        call_id="call_2",
        file_path=str(target),
        old_string="x = 1",
        new_string="x = 42",
    )

    assert target.read_text() == "x = 42\ny = 2\n"
    canonical = str(target.resolve())
    diff = "--- a/sample.py\n+++ b/sample.py\n@@ -1,2 +1,2 @@\n-x = 1\n+x = 42\n y = 2"
    output = {
        "file_path": canonical,
        "replacements": 1,
        "bytes_before": 12,
        "bytes_after": 13,
        "action": "edited",
        "diff": diff,
    }
    assert result == ToolResultMessage(
        content=output,
        id="call_2",
        display_text=f"Edited {canonical} (1 replacement, 12 → 13 bytes)\n{diff}",
    )


def test_edit_tool_replaces_all_occurrences(tmp_path: Path) -> None:
    """``replace_all=True`` replaces every occurrence."""
    target = tmp_path / "sample.txt"
    target.write_text("foo\nfoo\nfoo\n")

    ReadFileTool().call(call_id="call_1", file_path=str(target))
    result = EditFileTool().call(
        call_id="call_2",
        file_path=str(target),
        old_string="foo",
        new_string="bar",
        replace_all=True,
    )

    assert target.read_text() == "bar\nbar\nbar\n"
    output = result.content
    assert output["replacements"] == 3
    assert output["action"] == "edited"


def test_edit_tool_refuses_ambiguous_match_without_replace_all(
    tmp_path: Path,
) -> None:
    """Non-unique matches fail unless ``replace_all`` is true."""
    target = tmp_path / "sample.txt"
    target.write_text("foo\nfoo\n")

    ReadFileTool().call(call_id="call_1", file_path=str(target))
    result = EditFileTool().call(
        call_id="call_2",
        file_path=str(target),
        old_string="foo",
        new_string="bar",
    )

    assert target.read_text() == "foo\nfoo\n"
    output = (
        "Error while executing EditFileTool: "
        "old_string is not unique (occurs 2 times); "
        "use replace_all=True or expand context"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_2",
    )


def test_edit_tool_reports_missing_match(tmp_path: Path) -> None:
    """Failing to locate ``old_string`` surfaces as a tool error."""
    target = tmp_path / "sample.txt"
    target.write_text("hello\n")

    ReadFileTool().call(call_id="call_1", file_path=str(target))
    result = EditFileTool().call(
        call_id="call_2",
        file_path=str(target),
        old_string="goodbye",
        new_string="hello",
    )

    output = "Error while executing EditFileTool: old_string not found in file"
    assert result == ToolErrorMessage(
        content=output,
        id="call_2",
    )


def test_edit_tool_rejects_empty_old_string(tmp_path: Path) -> None:
    """Empty ``old_string`` is refused with a clear error."""
    target = tmp_path / "sample.txt"
    target.write_text("hello\n")

    ReadFileTool().call(call_id="call_1", file_path=str(target))
    result = EditFileTool().call(
        call_id="call_2",
        file_path=str(target),
        old_string="",
        new_string="x",
    )

    output = "Error while executing EditFileTool: old_string must not be empty"
    assert result == ToolErrorMessage(
        content=output,
        id="call_2",
    )


def test_edit_tool_rejects_identical_strings(tmp_path: Path) -> None:
    """Equal ``old_string``/``new_string`` is refused as a no-op."""
    target = tmp_path / "sample.txt"
    target.write_text("hello\n")

    ReadFileTool().call(call_id="call_1", file_path=str(target))
    result = EditFileTool().call(
        call_id="call_2",
        file_path=str(target),
        old_string="hello",
        new_string="hello",
    )

    output = (
        "Error while executing EditFileTool: "
        "old_string and new_string are identical; nothing to do"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_2",
    )


def test_edit_tool_rejects_relative_paths() -> None:
    """Relative paths surface as tool errors via the wrapper."""
    result = EditFileTool().call(
        call_id="call_1",
        file_path="relative.txt",
        old_string="a",
        new_string="b",
    )

    output = (
        "Error while executing EditFileTool: file_path must be absolute: relative.txt"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
    )


def test_edit_tool_reports_missing_file(tmp_path: Path) -> None:
    """Missing files surface as tool errors via the wrapper."""
    target = tmp_path / "missing.txt"

    result = EditFileTool().call(
        call_id="call_1",
        file_path=str(target),
        old_string="a",
        new_string="b",
    )

    output = f"Error while executing EditFileTool: File not found: {target}"
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
    )


def test_edit_tool_reports_directory_target(tmp_path: Path) -> None:
    """Directory targets surface as tool errors via the wrapper."""
    result = EditFileTool().call(
        call_id="call_1",
        file_path=str(tmp_path),
        old_string="a",
        new_string="b",
    )

    output = (
        f"Error while executing EditFileTool: Path is not a regular file: {tmp_path}"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
    )


def test_edit_tool_refuses_without_prior_read(tmp_path: Path) -> None:
    """Editing a file that was not read this session is refused."""
    target = tmp_path / "sample.txt"
    target.write_text("hello\n")

    result = EditFileTool().call(
        call_id="call_1",
        file_path=str(target),
        old_string="hello",
        new_string="world",
    )

    assert target.read_text() == "hello\n"
    output = (
        f"Error while executing EditFileTool: "
        f"File was not read this session; "
        f"read it first before editing: {target}"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
    )


def test_edit_tool_refuses_after_mtime_drift(tmp_path: Path) -> None:
    """Stale reads (mtime drifted since read) are rejected with a clear error."""
    target = tmp_path / "sample.txt"
    target.write_text("hello\n")

    original_mtime = target.stat().st_mtime
    ReadFileTool().call(call_id="call_1", file_path=str(target))

    drifted = original_mtime + 100.0
    os.utime(target, (drifted, drifted))

    result = EditFileTool().call(
        call_id="call_2",
        file_path=str(target),
        old_string="hello",
        new_string="world",
    )

    assert target.read_text() == "hello\n"
    output = (
        f"Error while executing EditFileTool: "
        f"File has changed on disk since it was read; "
        f"re-read before editing: {target}"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_2",
    )


def test_edit_tool_registers_post_edit_mtime(tmp_path: Path) -> None:
    """Successive edits without re-reading succeed via post-mutation record_read."""
    target = tmp_path / "sample.txt"
    target.write_text("first\nsecond\n")

    ReadFileTool().call(call_id="call_1", file_path=str(target))
    EditFileTool().call(
        call_id="call_2",
        file_path=str(target),
        old_string="first",
        new_string="FIRST",
    )
    result = EditFileTool().call(
        call_id="call_3",
        file_path=str(target),
        old_string="second",
        new_string="SECOND",
    )

    assert target.read_text() == "FIRST\nSECOND\n"
    assert result.content["replacements"] == 1


def test_tool_raises_when_harness_dir_missing() -> None:
    """Instantiating a tool without harness assets raises FileNotFoundError."""

    class MissingTool(BaseTool):
        def _call(self, **kwargs: Any) -> str:
            return ""

    with pytest.raises(FileNotFoundError, match="harness/tools/missing_tool"):
        MissingTool()


def test_tool_name_property_returns_class_name() -> None:
    """``name`` property returns the python class name."""
    assert ShellCommandTool().name == "ShellCommandTool"


def test_tool_instance_loads_description_from_harness_dir() -> None:
    """Tool instance populates ``self.description`` from parsed ``description.md``."""
    tool = ShellCommandTool()
    harness_root = Path("harness")
    expected_file = DynamicMarkdownFile(
        harness_root / "tools/shell_command_tool/description.md"
    )
    expected_file.parse(tool=tool)
    assert tool.description == expected_file.content


def test_tool_instance_loads_input_schema_from_harness_dir() -> None:
    """Tool instance populates ``self.input_schema`` from ``input_schema.yml``."""
    raw = yaml.safe_load(
        Path("harness/tools/shell_command_tool/input_schema.yml").read_text()
    )
    expected = {"type": "object", "additionalProperties": False, **raw}
    assert ShellCommandTool().input_schema == expected


def test_tool_instance_schema_carries_loaded_attrs() -> None:
    """``schema()`` returns a ToolSchema built from harness-loaded attributes."""
    tool = ShellCommandTool()
    harness_root = Path("harness")
    tool_dir = harness_root / "tools/shell_command_tool"
    expected_description_file = DynamicMarkdownFile(tool_dir / "description.md")
    expected_description_file.parse(tool=tool)
    expected_input_schema_raw = yaml.safe_load(
        (tool_dir / "input_schema.yml").read_text()
    )
    expected_input_schema = {
        "type": "object",
        "additionalProperties": False,
        **expected_input_schema_raw,
    }

    assert tool.schema() == ToolSchema(
        name="ShellCommandTool",
        description=expected_description_file.content,
        input_schema=expected_input_schema,
    )


# — SearchWebTool —————————————————————————————————————————————————————————


def _ddgs_hit(href: str, title: str, body: str) -> dict[str, str]:
    """Build a ddgs-shaped raw hit dict.

    Args:
        href: result URL.
        title: result title.
        body: result excerpt.

    Returns:
        A dict using ddgs' native ``href``/``title``/``body`` keys.
    """
    return {"href": href, "title": title, "body": body}


def test_search_web_tool_happy_path() -> None:
    """Happy path returns normalized results and a numbered formatted body."""
    raw = [
        _ddgs_hit("https://pytorch.org/docs/", "PyTorch docs", "Docs excerpt."),
        _ddgs_hit(
            "https://pytorch.org/tutorials/",
            "PyTorch tutorials",
            "Tut excerpt.",
        ),
    ]
    with patch("arancio.core.tools.web.search.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = raw
        result = SearchWebTool().call(call_id="call_1", query="pytorch")

    expected_output = {
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
        content=expected_output,
        id="call_1",
        display_text=expected_content,
    )


def test_search_web_tool_clamps_num_results_to_max() -> None:
    """num_results above the cap is clamped before being passed to ddgs."""
    with patch("arancio.core.tools.web.search.DDGS") as mock_ddgs:
        text = mock_ddgs.return_value.__enter__.return_value.text
        text.return_value = []
        SearchWebTool().call(call_id="call_1", query="x", num_results=50)

    text.assert_called_once_with("x", max_results=20)


def test_search_web_tool_clamps_timeout_to_max() -> None:
    """Timeout above the cap is clamped before being passed to DDGS()."""
    with patch("arancio.core.tools.web.search.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = []
        SearchWebTool().call(call_id="call_1", query="x", timeout=9000)

    mock_ddgs.assert_called_once_with(timeout=300)


def test_search_web_tool_uses_defaults_when_args_missing() -> None:
    """When num_results/timeout are omitted the class defaults are used."""
    with patch("arancio.core.tools.web.search.DDGS") as mock_ddgs:
        text = mock_ddgs.return_value.__enter__.return_value.text
        text.return_value = []
        SearchWebTool().call(call_id="call_1", query="x")

    mock_ddgs.assert_called_once_with(timeout=60)
    text.assert_called_once_with("x", max_results=10)


def test_search_web_tool_empty_results_formats_no_results_footer() -> None:
    """Zero hits produces a ``[no results]`` content body and a success message."""
    with patch("arancio.core.tools.web.search.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = []
        result = SearchWebTool().call(call_id="call_1", query="nothing matches")

    assert result == ToolResultMessage(
        content={"query": "nothing matches", "results": [], "timed_out": False},
        id="call_1",
        display_text="[no results]",
    )


def test_search_web_tool_timeout_returns_tool_error_message() -> None:
    """``TimeoutException`` surfaces as a ToolErrorMessage with timed_out=True."""
    from ddgs.exceptions import TimeoutException

    with patch("arancio.core.tools.web.search.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.side_effect = (
            TimeoutException("ddgs timed out")
        )
        result = SearchWebTool().call(call_id="call_1", query="slow", timeout=1)

    assert result == ToolErrorMessage(
        content={"query": "slow", "results": [], "timed_out": True},
        id="call_1",
        display_text="[no results]\n[timed out]",
    )


def test_search_web_tool_arbitrary_exception_wraps_as_error_string() -> None:
    """Non-timeout failures propagate and are wrapped by BaseTool.call."""
    from ddgs.exceptions import DDGSException

    with patch("arancio.core.tools.web.search.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.side_effect = DDGSException(
            "rate limited"
        )
        result = SearchWebTool().call(call_id="call_1", query="x")

    assert isinstance(result, ToolErrorMessage)
    assert result.id == "call_1"
    assert "rate limited" in result.display_text
    assert result.display_text.startswith("Error while executing SearchWebTool:")
    assert result.content == result.display_text


def test_search_web_tool_handles_missing_fields_in_raw_hits() -> None:
    """Raw hits missing href/title/body keys default to empty strings."""
    with patch("arancio.core.tools.web.search.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [{}]
        result = SearchWebTool().call(call_id="call_1", query="x")

    assert result.content["results"] == [{"url": "", "title": "", "excerpt": ""}]


# — FetchWebTool —————————————————————————————————————————————————————————————


def _fetch_tool_with_answer(answer: str):
    """Build a FetchWebTool with a mocked injected client returning an answer.

    Args:
        answer: text the mocked summarization client returns as its finalized
            assistant message.

    Returns:
        A tuple ``(tool, client)`` where ``client`` is the mocked summarization
        client so tests can inspect its calls.
    """
    client = MagicMock()
    client.send_request.return_value = [AssistantMessage(content=answer)]
    return FetchWebTool(client=client), client


def test_fetch_web_tool_happy_path() -> None:
    """Happy path returns the answer plus metadata and a source footer."""
    tool, client = _fetch_tool_with_answer("The page covers X.")

    with (
        patch("arancio.core.tools.web.fetch.trafilatura") as traf,
        patch("arancio.core.tools.web.fetch.use_config"),
    ):
        traf.fetch_url.return_value = "<html>...</html>"
        traf.extract.return_value = "# Heading\nbody"
        traf.extract_metadata.return_value = SimpleNamespace(
            title="Title", url="https://x/final"
        )
        result = tool.call(
            call_id="call_1",
            url="https://x",
            query="what does it cover?",
        )

    output = result.content
    assert output["url"] == "https://x/final"
    assert output["title"] == "Title"
    assert output["query"] == "what does it cover?"
    assert output["answer"] == "The page covers X."
    assert output["content_type"] is None
    assert output["truncated"] is False
    assert "retrieved_at" in output
    assert isinstance(result, ToolResultMessage)
    assert result.display_text == "The page covers X.\n\n[Title — https://x/final]"

    kwargs = client.build_request.call_args.kwargs
    assert kwargs["tools"] == []
    assert "Task: what does it cover?" in kwargs["messages"][0].content


def test_fetch_web_tool_rejects_non_http_url() -> None:
    """Non-http(s) URLs are rejected before any fetch."""
    tool = FetchWebTool(client=MagicMock())

    result = tool.call(call_id="call_1", url="ftp://x", query="q")

    assert isinstance(result, ToolErrorMessage)
    assert "url must be an http(s) URL" in result.display_text


def test_fetch_web_tool_fetch_failure_is_error() -> None:
    """A None download surfaces as a tool error."""
    tool = FetchWebTool(client=MagicMock())

    with (
        patch("arancio.core.tools.web.fetch.trafilatura") as traf,
        patch("arancio.core.tools.web.fetch.use_config"),
    ):
        traf.fetch_url.return_value = None
        result = tool.call(call_id="call_1", url="https://x", query="q")

    assert isinstance(result, ToolErrorMessage)
    assert "failed to fetch https://x" in result.display_text


def test_fetch_web_tool_empty_extraction_is_error() -> None:
    """No extractable content surfaces as a tool error."""
    tool = FetchWebTool(client=MagicMock())

    with (
        patch("arancio.core.tools.web.fetch.trafilatura") as traf,
        patch("arancio.core.tools.web.fetch.use_config"),
    ):
        traf.fetch_url.return_value = "<html></html>"
        traf.extract.return_value = None
        result = tool.call(call_id="call_1", url="https://x", query="q")

    assert isinstance(result, ToolErrorMessage)
    assert "no readable content extracted from https://x" in result.display_text


def test_fetch_web_tool_truncates_long_content() -> None:
    """Content longer than the cap is truncated before summarization."""
    tool, client = _fetch_tool_with_answer("ok")
    long_text = "a" * (FetchWebTool._max_content_chars + 10)

    with (
        patch("arancio.core.tools.web.fetch.trafilatura") as traf,
        patch("arancio.core.tools.web.fetch.use_config"),
    ):
        traf.fetch_url.return_value = "<html></html>"
        traf.extract.return_value = long_text
        traf.extract_metadata.return_value = SimpleNamespace(title="T", url="https://x")
        result = tool.call(call_id="call_1", url="https://x", query="q")

    assert result.content["truncated"] is True
    sent = client.build_request.call_args.kwargs["messages"][0].content
    assert "a" * FetchWebTool._max_content_chars in sent
    assert "a" * (FetchWebTool._max_content_chars + 1) not in sent


def test_fetch_web_tool_clamps_timeout_to_max() -> None:
    """Timeout above the cap is clamped into the trafilatura config."""
    tool, _ = _fetch_tool_with_answer("ok")

    with (
        patch("arancio.core.tools.web.fetch.trafilatura") as traf,
        patch("arancio.core.tools.web.fetch.use_config") as mock_use_config,
    ):
        cfg = mock_use_config.return_value
        traf.fetch_url.return_value = "<html></html>"
        traf.extract.return_value = "body"
        traf.extract_metadata.return_value = SimpleNamespace(
            title=None, url="https://x"
        )
        tool.call(call_id="call_1", url="https://x", query="q", timeout=9000)

    cfg.set.assert_called_once_with("DEFAULT", "DOWNLOAD_TIMEOUT", "300")


def test_fetch_web_tool_empty_answer_is_error() -> None:
    """An empty summarization answer surfaces as a tool error."""
    tool, _ = _fetch_tool_with_answer("")

    with (
        patch("arancio.core.tools.web.fetch.trafilatura") as traf,
        patch("arancio.core.tools.web.fetch.use_config"),
    ):
        traf.fetch_url.return_value = "<html></html>"
        traf.extract.return_value = "body"
        traf.extract_metadata.return_value = SimpleNamespace(
            title=None, url="https://x"
        )
        result = tool.call(call_id="call_1", url="https://x", query="q")

    assert isinstance(result, ToolErrorMessage)
    assert "summarization produced no answer" in result.display_text


def test_fetch_web_tool_llm_failure_is_error() -> None:
    """A summarization client exception surfaces as a tool error."""
    client = MagicMock()
    client.send_request.side_effect = RuntimeError("llm down")
    tool = FetchWebTool(client=client)

    with (
        patch("arancio.core.tools.web.fetch.trafilatura") as traf,
        patch("arancio.core.tools.web.fetch.use_config"),
    ):
        traf.fetch_url.return_value = "<html></html>"
        traf.extract.return_value = "body"
        traf.extract_metadata.return_value = SimpleNamespace(
            title=None, url="https://x"
        )
        result = tool.call(call_id="call_1", url="https://x", query="q")

    assert isinstance(result, ToolErrorMessage)
    assert "llm down" in result.display_text
