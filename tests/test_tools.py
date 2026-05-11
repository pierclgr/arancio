"""Tests for tool execution helpers."""

import os
import subprocess
from pathlib import Path
from typing import Any, ClassVar, List
from unittest.mock import patch

import pytest
import yaml
from dynamic_markdown.types.files.base import DynamicMarkdownFile

from codo.parsers.tool_result.base import BaseToolResultParser
from codo.tools.base import BaseTool
from codo.tools.edit_tool import EditTool
from codo.tools.grep_tool import GrepTool
from codo.tools.powershell_command import PowershellCommandTool
from codo.tools.read_tool import ReadTool
from codo.tools.session import default_session
from codo.tools.write_tool import WriteTool
from codo.types.messages import ToolErrorMessage, ToolResultMessage
from codo.types.tools import ToolSchema


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
            output=output,
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
        output=output,
    )


@patch("codo.tools.powershell_command.subprocess.run")
@patch("codo.tools.powershell_command.shutil.which")
def test_powershell_command_prefers_windows_powershell(
    which_mock,
    run_mock,
) -> None:
    """PowerShell tool prefers powershell.exe when it is available."""
    which_mock.side_effect = lambda name: (
        "powershell.exe" if name == "powershell.exe" else None
    )
    run_mock.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="ok\n",
        stderr="",
    )

    result = PowershellCommandTool().call(
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
        content="ok",
        id="call_1",
        output=output,
    )


@patch("codo.tools.powershell_command.subprocess.run")
@patch("codo.tools.powershell_command.shutil.which")
def test_powershell_command_falls_back_to_pwsh(which_mock, run_mock) -> None:
    """PowerShell tool falls back to pwsh when powershell.exe is unavailable."""
    which_mock.side_effect = lambda name: "pwsh" if name == "pwsh" else None
    run_mock.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="ok\n",
        stderr="",
    )

    PowershellCommandTool().call(call_id="call_1", command="Write-Output ok")

    command_args = run_mock.call_args.args[0]
    assert command_args[0] == "pwsh"
    assert command_args[-1] == "Write-Output ok"


@patch("codo.tools.powershell_command.subprocess.run")
@patch("codo.tools.powershell_command.shutil.which")
def test_powershell_command_reports_missing_host(which_mock, run_mock) -> None:
    """Missing PowerShell hosts are reported as tool errors."""
    which_mock.return_value = None

    result = PowershellCommandTool().call(call_id="call_1", command="Get-ChildItem")

    run_mock.assert_not_called()
    output = (
        "Error while executing PowershellCommandTool: "
        "PowerShell host not found: powershell.exe or pwsh"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        output=output,
    )


@patch("codo.tools.powershell_command.subprocess.run")
@patch("codo.tools.powershell_command.shutil.which")
def test_powershell_command_reports_timeout(which_mock, run_mock) -> None:
    """Timed-out PowerShell commands return partial output as an error."""
    which_mock.return_value = "powershell.exe"
    run_mock.side_effect = subprocess.TimeoutExpired(
        cmd=[],
        timeout=1,
        output=b"partial\n",
        stderr=b"slow\n",
    )

    result = PowershellCommandTool().call(
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
        content="partial\nslow\n[timed out]\n[exit code -1]",
        id="call_1",
        output=output,
    )


@patch("codo.tools.powershell_command.subprocess.run")
@patch("codo.tools.powershell_command.shutil.which")
def test_powershell_command_truncates_long_output(which_mock, run_mock) -> None:
    """PowerShell command streams are truncated independently."""
    which_mock.return_value = "powershell.exe"
    run_mock.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="a" * 30_001,
        stderr="",
    )

    output = PowershellCommandTool()._call(command="Write-Output long")

    assert output["stdout"].startswith("a" * 30_000)
    assert "1 chars truncated" in output["stdout"]
    assert output["truncated"] is True


def test_read_tool_reads_full_file_with_line_numbers(tmp_path: Path) -> None:
    """ReadTool returns each line prefixed with its 1-indexed line number."""
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\ngamma\n")

    result = ReadTool().call(call_id="call_1", file_path=str(target))

    expected_block = "     1\talpha\n     2\tbeta\n     3\tgamma"
    output = {
        "content": expected_block,
        "start_line": 1,
        "end_line": 3,
        "total_lines": 3,
        "truncated_lines": 0,
    }
    assert result == ToolResultMessage(
        content=f"{expected_block}\n[lines 1-3 of 3]",
        id="call_1",
        output=output,
    )


def test_read_tool_slices_with_offset_and_limit(tmp_path: Path) -> None:
    """ReadTool respects offset and limit while preserving line numbers."""
    target = tmp_path / "sample.txt"
    target.write_text("\n".join(f"line-{i}" for i in range(1, 6)) + "\n")

    result = ReadTool().call(
        call_id="call_1",
        file_path=str(target),
        offset=2,
        limit=2,
    )

    expected_block = "     2\tline-2\n     3\tline-3"
    output = {
        "content": expected_block,
        "start_line": 2,
        "end_line": 3,
        "total_lines": 5,
        "truncated_lines": 0,
    }
    assert result == ToolResultMessage(
        content=f"{expected_block}\n[lines 2-3 of 5]",
        id="call_1",
        output=output,
    )


def test_read_tool_truncates_long_lines(tmp_path: Path) -> None:
    """Lines exceeding the per-line cap are truncated with a marker."""
    long_line = "a" * (ReadTool._max_line_chars + 25)
    target = tmp_path / "sample.txt"
    target.write_text(long_line + "\n")

    result = ReadTool().call(call_id="call_1", file_path=str(target))

    capped = "a" * ReadTool._max_line_chars + ReadTool._line_truncation_marker
    expected_block = f"     1\t{capped}"
    output = {
        "content": expected_block,
        "start_line": 1,
        "end_line": 1,
        "total_lines": 1,
        "truncated_lines": 1,
    }
    assert result == ToolResultMessage(
        content=f"{expected_block}\n[lines 1-1 of 1]\n[1 long lines truncated]",
        id="call_1",
        output=output,
    )


def test_read_tool_handles_empty_file(tmp_path: Path) -> None:
    """Empty files return an empty content block and zero counts."""
    target = tmp_path / "empty.txt"
    target.write_text("")

    result = ReadTool().call(call_id="call_1", file_path=str(target))

    output = {
        "content": "",
        "start_line": 0,
        "end_line": 0,
        "total_lines": 0,
        "truncated_lines": 0,
    }
    assert result == ToolResultMessage(
        content="[empty file]",
        id="call_1",
        output=output,
    )


def test_read_tool_rejects_relative_paths() -> None:
    """Relative paths surface as tool errors via the wrapper."""
    result = ReadTool().call(call_id="call_1", file_path="relative.txt")

    output = "Error while executing ReadTool: file_path must be absolute: relative.txt"
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        output=output,
    )


def test_read_tool_reports_missing_file(tmp_path: Path) -> None:
    """Missing files surface as tool errors via the wrapper."""
    target = tmp_path / "missing.txt"

    result = ReadTool().call(call_id="call_1", file_path=str(target))

    output = f"Error while executing ReadTool: File not found: {target}"
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        output=output,
    )


def test_read_tool_reports_directory_path(tmp_path: Path) -> None:
    """Directory paths surface as tool errors via the wrapper."""
    result = ReadTool().call(call_id="call_1", file_path=str(tmp_path))

    output = f"Error while executing ReadTool: Path is not a regular file: {tmp_path}"
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        output=output,
    )


def test_read_tool_registers_read_with_session(tmp_path: Path) -> None:
    """Successful reads register the file in the shared tool session."""
    target = tmp_path / "sample.txt"
    target.write_text("hello\n")

    ReadTool().call(call_id="call_1", file_path=str(target))

    canonical = str(target.resolve())
    assert default_session.is_known(canonical) is True
    assert default_session.is_fresh(canonical, target.stat().st_mtime) is True


def test_write_tool_creates_new_file(tmp_path: Path) -> None:
    """Writing to a path that does not exist creates the file."""
    target = tmp_path / "new.txt"

    result = WriteTool().call(
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
        content=f"Created {canonical} (12 bytes, 2 lines)",
        id="call_1",
        output=output,
    )


def test_write_tool_overwrites_after_read(tmp_path: Path) -> None:
    """Pre-reading the file unlocks an overwrite of an existing target."""
    target = tmp_path / "sample.txt"
    target.write_text("old")

    ReadTool().call(call_id="call_1", file_path=str(target))
    result = WriteTool().call(
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
        content=f"Overwrote {canonical} (11 bytes, 1 lines)",
        id="call_2",
        output=output,
    )


def test_write_tool_refuses_overwrite_without_prior_read(tmp_path: Path) -> None:
    """Existing files that were not read this session cannot be overwritten."""
    target = tmp_path / "sample.txt"
    target.write_text("untouched")

    result = WriteTool().call(
        call_id="call_1",
        file_path=str(target),
        content="should not land",
    )

    assert target.read_text() == "untouched"
    output = (
        f"Error while executing WriteTool: "
        f"File exists but was not read this session; "
        f"read it first before overwriting: {target}"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        output=output,
    )


def test_write_tool_refuses_overwrite_after_mtime_drift(tmp_path: Path) -> None:
    """Stale reads (mtime drifted since read) are rejected with a clear error."""
    target = tmp_path / "sample.txt"
    target.write_text("old")

    original_mtime = target.stat().st_mtime
    ReadTool().call(call_id="call_1", file_path=str(target))

    drifted = original_mtime + 100.0
    os.utime(target, (drifted, drifted))

    result = WriteTool().call(
        call_id="call_2",
        file_path=str(target),
        content="should not land",
    )

    assert target.read_text() == "old"
    output = (
        f"Error while executing WriteTool: "
        f"File has changed on disk since it was read; "
        f"re-read before overwriting: {target}"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_2",
        output=output,
    )


def test_write_tool_rejects_relative_paths() -> None:
    """Relative paths surface as tool errors via the wrapper."""
    result = WriteTool().call(
        call_id="call_1",
        file_path="relative.txt",
        content="x",
    )

    output = "Error while executing WriteTool: file_path must be absolute: relative.txt"
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        output=output,
    )


def test_write_tool_reports_missing_parent_dir(tmp_path: Path) -> None:
    """Missing parent directories surface as tool errors via the wrapper."""
    target = tmp_path / "missing_dir" / "file.txt"

    result = WriteTool().call(
        call_id="call_1",
        file_path=str(target),
        content="x",
    )

    output = (
        f"Error while executing WriteTool: Parent directory not found: {target.parent}"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        output=output,
    )


def test_write_tool_reports_directory_target(tmp_path: Path) -> None:
    """Directory targets surface as tool errors via the wrapper."""
    result = WriteTool().call(
        call_id="call_1",
        file_path=str(tmp_path),
        content="x",
    )

    output = f"Error while executing WriteTool: Path is not a regular file: {tmp_path}"
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        output=output,
    )


def test_write_tool_handles_empty_content(tmp_path: Path) -> None:
    """Empty content creates a zero-byte file with zero lines."""
    target = tmp_path / "empty.txt"

    result = WriteTool().call(
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
        content=f"Created {canonical} (0 bytes, 0 lines)",
        id="call_1",
        output=output,
    )


def test_edit_tool_replaces_unique_match(tmp_path: Path) -> None:
    """A unique substring is replaced and the file content updated on disk."""
    target = tmp_path / "sample.py"
    target.write_text("x = 1\ny = 2\n")

    ReadTool().call(call_id="call_1", file_path=str(target))
    result = EditTool().call(
        call_id="call_2",
        file_path=str(target),
        old_string="x = 1",
        new_string="x = 42",
    )

    assert target.read_text() == "x = 42\ny = 2\n"
    canonical = str(target.resolve())
    output = {
        "file_path": canonical,
        "replacements": 1,
        "bytes_before": 12,
        "bytes_after": 13,
        "action": "edited",
    }
    assert result == ToolResultMessage(
        content=f"Edited {canonical} (1 replacement, 12 → 13 bytes)",
        id="call_2",
        output=output,
    )


def test_edit_tool_replaces_all_occurrences(tmp_path: Path) -> None:
    """``replace_all=True`` replaces every occurrence."""
    target = tmp_path / "sample.txt"
    target.write_text("foo\nfoo\nfoo\n")

    ReadTool().call(call_id="call_1", file_path=str(target))
    result = EditTool().call(
        call_id="call_2",
        file_path=str(target),
        old_string="foo",
        new_string="bar",
        replace_all=True,
    )

    assert target.read_text() == "bar\nbar\nbar\n"
    output = result.output
    assert output["replacements"] == 3
    assert output["action"] == "edited"


def test_edit_tool_refuses_ambiguous_match_without_replace_all(
    tmp_path: Path,
) -> None:
    """Non-unique matches fail unless ``replace_all`` is true."""
    target = tmp_path / "sample.txt"
    target.write_text("foo\nfoo\n")

    ReadTool().call(call_id="call_1", file_path=str(target))
    result = EditTool().call(
        call_id="call_2",
        file_path=str(target),
        old_string="foo",
        new_string="bar",
    )

    assert target.read_text() == "foo\nfoo\n"
    output = (
        "Error while executing EditTool: "
        "old_string is not unique (occurs 2 times); "
        "use replace_all=True or expand context"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_2",
        output=output,
    )


def test_edit_tool_reports_missing_match(tmp_path: Path) -> None:
    """Failing to locate ``old_string`` surfaces as a tool error."""
    target = tmp_path / "sample.txt"
    target.write_text("hello\n")

    ReadTool().call(call_id="call_1", file_path=str(target))
    result = EditTool().call(
        call_id="call_2",
        file_path=str(target),
        old_string="goodbye",
        new_string="hello",
    )

    output = "Error while executing EditTool: old_string not found in file"
    assert result == ToolErrorMessage(
        content=output,
        id="call_2",
        output=output,
    )


def test_edit_tool_rejects_empty_old_string(tmp_path: Path) -> None:
    """Empty ``old_string`` is refused with a clear error."""
    target = tmp_path / "sample.txt"
    target.write_text("hello\n")

    ReadTool().call(call_id="call_1", file_path=str(target))
    result = EditTool().call(
        call_id="call_2",
        file_path=str(target),
        old_string="",
        new_string="x",
    )

    output = "Error while executing EditTool: old_string must not be empty"
    assert result == ToolErrorMessage(
        content=output,
        id="call_2",
        output=output,
    )


def test_edit_tool_rejects_identical_strings(tmp_path: Path) -> None:
    """Equal ``old_string``/``new_string`` is refused as a no-op."""
    target = tmp_path / "sample.txt"
    target.write_text("hello\n")

    ReadTool().call(call_id="call_1", file_path=str(target))
    result = EditTool().call(
        call_id="call_2",
        file_path=str(target),
        old_string="hello",
        new_string="hello",
    )

    output = (
        "Error while executing EditTool: "
        "old_string and new_string are identical; nothing to do"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_2",
        output=output,
    )


def test_edit_tool_rejects_relative_paths() -> None:
    """Relative paths surface as tool errors via the wrapper."""
    result = EditTool().call(
        call_id="call_1",
        file_path="relative.txt",
        old_string="a",
        new_string="b",
    )

    output = "Error while executing EditTool: file_path must be absolute: relative.txt"
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        output=output,
    )


def test_edit_tool_reports_missing_file(tmp_path: Path) -> None:
    """Missing files surface as tool errors via the wrapper."""
    target = tmp_path / "missing.txt"

    result = EditTool().call(
        call_id="call_1",
        file_path=str(target),
        old_string="a",
        new_string="b",
    )

    output = f"Error while executing EditTool: File not found: {target}"
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        output=output,
    )


def test_edit_tool_reports_directory_target(tmp_path: Path) -> None:
    """Directory targets surface as tool errors via the wrapper."""
    result = EditTool().call(
        call_id="call_1",
        file_path=str(tmp_path),
        old_string="a",
        new_string="b",
    )

    output = f"Error while executing EditTool: Path is not a regular file: {tmp_path}"
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        output=output,
    )


def test_edit_tool_refuses_without_prior_read(tmp_path: Path) -> None:
    """Editing a file that was not read this session is refused."""
    target = tmp_path / "sample.txt"
    target.write_text("hello\n")

    result = EditTool().call(
        call_id="call_1",
        file_path=str(target),
        old_string="hello",
        new_string="world",
    )

    assert target.read_text() == "hello\n"
    output = (
        f"Error while executing EditTool: "
        f"File was not read this session; "
        f"read it first before editing: {target}"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_1",
        output=output,
    )


def test_edit_tool_refuses_after_mtime_drift(tmp_path: Path) -> None:
    """Stale reads (mtime drifted since read) are rejected with a clear error."""
    target = tmp_path / "sample.txt"
    target.write_text("hello\n")

    original_mtime = target.stat().st_mtime
    ReadTool().call(call_id="call_1", file_path=str(target))

    drifted = original_mtime + 100.0
    os.utime(target, (drifted, drifted))

    result = EditTool().call(
        call_id="call_2",
        file_path=str(target),
        old_string="hello",
        new_string="world",
    )

    assert target.read_text() == "hello\n"
    output = (
        f"Error while executing EditTool: "
        f"File has changed on disk since it was read; "
        f"re-read before editing: {target}"
    )
    assert result == ToolErrorMessage(
        content=output,
        id="call_2",
        output=output,
    )


def test_edit_tool_registers_post_edit_mtime(tmp_path: Path) -> None:
    """Successive edits without re-reading succeed via post-mutation record_read."""
    target = tmp_path / "sample.txt"
    target.write_text("first\nsecond\n")

    ReadTool().call(call_id="call_1", file_path=str(target))
    EditTool().call(
        call_id="call_2",
        file_path=str(target),
        old_string="first",
        new_string="FIRST",
    )
    result = EditTool().call(
        call_id="call_3",
        file_path=str(target),
        old_string="second",
        new_string="SECOND",
    )

    assert target.read_text() == "FIRST\nSECOND\n"
    assert result.output["replacements"] == 1


def test_tool_raises_when_harness_dir_missing() -> None:
    """Instantiating a tool without harness assets raises FileNotFoundError."""

    class MissingTool(BaseTool):
        def _call(self, **kwargs: Any) -> str:
            return ""

    with pytest.raises(FileNotFoundError, match="harness/tools/missing_tool"):
        MissingTool()


def test_tool_name_property_returns_class_name() -> None:
    """``name`` property returns the python class name."""
    assert PowershellCommandTool().name == "PowershellCommandTool"


def test_tool_instance_loads_description_from_harness_dir() -> None:
    """Tool instance populates ``self.description`` from parsed ``description.md``."""
    tool = PowershellCommandTool()
    harness_root = Path("harness")
    expected_file = DynamicMarkdownFile(
        harness_root / "tools/powershell_command_tool/description.md"
    )
    expected_file.parse(base_dir=harness_root, tool=tool)
    assert tool.description == expected_file.content


def test_tool_instance_loads_input_schema_from_harness_dir() -> None:
    """Tool instance populates ``self.input_schema`` from ``input_schema.yml``."""
    raw = yaml.safe_load(
        Path("harness/tools/powershell_command_tool/input_schema.yml").read_text()
    )
    expected = {"type": "object", "additionalProperties": False, **raw}
    assert PowershellCommandTool().input_schema == expected


def test_tool_instance_schema_carries_loaded_attrs() -> None:
    """``schema()`` returns a ToolSchema built from harness-loaded attributes."""
    tool = PowershellCommandTool()
    harness_root = Path("harness")
    tool_dir = harness_root / "tools/powershell_command_tool"
    expected_description_file = DynamicMarkdownFile(tool_dir / "description.md")
    expected_description_file.parse(base_dir=harness_root, tool=tool)
    expected_input_schema_raw = yaml.safe_load(
        (tool_dir / "input_schema.yml").read_text()
    )
    expected_input_schema = {
        "type": "object",
        "additionalProperties": False,
        **expected_input_schema_raw,
    }

    assert tool.schema() == ToolSchema(
        name="PowershellCommandTool",
        description=expected_description_file.content,
        input_schema=expected_input_schema,
    )


# — GrepTool ——————————————————————————————————————————————————————————


def test_grep_tool_files_with_matches_default(tmp_path: Path) -> None:
    """Default mode returns only matching file paths."""
    target = tmp_path / "greptest"
    target.mkdir()
    (target / "a.py").write_text("def foo():\n    pass\n")
    (target / "b.py").write_text("x = 1\n")
    (target / "c.txt").write_text("not python\n")

    result = GrepTool().call(
        call_id="call_1",
        pattern=r"def\s+\w+",
        path=str(target),
        output_mode="files_with_matches",
    )

    canonical_a = str((target / "a.py").resolve())
    output = result.output
    assert output["output_mode"] == "files_with_matches"
    assert canonical_a in output["matches"]
    assert output["total_matches"] == 1
    assert output["truncated"] is False
    assert result == ToolResultMessage(
        content=f"{canonical_a}\n[1 file matched]",
        id="call_1",
        output=output,
    )


def test_grep_tool_content_mode(tmp_path: Path) -> None:
    """Content mode returns matching lines with file, line number and content."""
    target = tmp_path / "greptest"
    target.mkdir()
    (target / "app.py").write_text("def alpha():\n    pass\n\ndef beta():\n    pass\n")

    result = GrepTool().call(
        call_id="call_1",
        pattern=r"def\s+\w+",
        path=str(target),
        output_mode="content",
    )

    canonical = str((target / "app.py").resolve())
    output = result.output
    assert output["output_mode"] == "content"
    assert output["matches"] == [
        {"file": canonical, "line": 1, "content": "def alpha():", "is_context": False},
        {"file": canonical, "line": 4, "content": "def beta():", "is_context": False},
    ]
    assert output["total_matches"] == 2


def test_grep_tool_count_mode(tmp_path: Path) -> None:
    """Count mode returns match counts per file."""
    target = tmp_path / "greptest"
    target.mkdir()
    (target / "f.py").write_text("foo\nfoo\nfoo\nbar\nfoo\n")

    result = GrepTool().call(
        call_id="call_1",
        pattern="foo",
        path=str(target),
        output_mode="count",
    )

    canonical = str((target / "f.py").resolve())
    output = result.output
    assert output["output_mode"] == "count"
    assert output["matches"] == [
        {"file": canonical, "count": 4},
    ]
    assert output["total_matches"] == 1


def test_grep_tool_case_insensitive(tmp_path: Path) -> None:
    """Case insensitive flag matches mixed-case content."""
    target = tmp_path / "greptest"
    target.mkdir()
    (target / "f.py").write_text("Hello World\n")

    result = GrepTool().call(
        call_id="call_1",
        pattern="hello",
        path=str(target),
        output_mode="files_with_matches",
        i=True,
    )

    assert len(result.output["matches"]) == 1


def test_grep_tool_glob_filter(tmp_path: Path) -> None:
    """Glob filter restricts search to matching filenames only."""
    target = tmp_path / "greptest"
    target.mkdir()
    (target / "a.py").write_text("hello\n")
    (target / "b.txt").write_text("hello\n")

    result = GrepTool().call(
        call_id="call_1",
        pattern="hello",
        path=str(target),
        output_mode="files_with_matches",
        glob="*.py",
    )

    output = result.output
    assert len(output["matches"]) == 1
    assert output["matches"][0].endswith("a.py")


def test_grep_tool_file_type_filter(tmp_path: Path) -> None:
    """File type filter restricts search to the given language."""
    target = tmp_path / "greptest"
    target.mkdir()
    (target / "a.py").write_text("hello\n")
    (target / "b.js").write_text("hello\n")

    result = GrepTool().call(
        call_id="call_1",
        pattern="hello",
        path=str(target),
        output_mode="files_with_matches",
        file_type="py",
    )

    output = result.output
    assert len(output["matches"]) == 1
    assert output["matches"][0].endswith("a.py")


def test_grep_tool_head_limit_caps_output(tmp_path: Path) -> None:
    """Head limit truncates output and sets the truncated flag."""
    target = tmp_path / "greptest"
    target.mkdir()
    content = "\n".join(f"line-{i}" for i in range(20))
    (target / "f.txt").write_text(content + "\n")

    result = GrepTool().call(
        call_id="call_1",
        pattern=r"line-\d+",
        path=str(target),
        output_mode="content",
        head_limit=5,
    )

    output = result.output
    assert len(output["matches"]) == 5
    assert output["total_matches"] == 20
    assert output["truncated"] is True
    assert "truncated" in result.content


def test_grep_tool_context_lines(tmp_path: Path) -> None:
    """Context lines are returned with the ``is_context`` flag set."""
    target = tmp_path / "greptest"
    target.mkdir()
    (target / "f.txt").write_text("before\nmatch\nmiddle\nafter\n")

    result = GrepTool().call(
        call_id="call_1",
        pattern="match",
        path=str(target),
        output_mode="content",
        A=1,
        B=1,
    )

    output = result.output
    assert len(output["matches"]) == 3
    contexts = [m for m in output["matches"] if m["is_context"]]
    matches = [m for m in output["matches"] if not m["is_context"]]
    assert len(contexts) == 2
    assert len(matches) == 1
    assert matches[0]["content"] == "match"


def test_grep_tool_zero_matches(tmp_path: Path) -> None:
    """Zero matches returns an empty result with exit code 1."""
    target = tmp_path / "greptest"
    target.mkdir()
    (target / "f.txt").write_text("nothing here\n")

    result = GrepTool().call(
        call_id="call_1",
        pattern="nonesuch",
        path=str(target),
        output_mode="content",
    )

    output = result.output
    assert output["matches"] == []
    assert output["total_matches"] == 0
    assert output["exit_code"] == 1


def test_grep_tool_escaped_literal_braces(tmp_path: Path) -> None:
    """Literal braces are matched when properly escaped."""
    target = tmp_path / "greptest"
    target.mkdir()
    (target / "f.txt").write_text("interface{}\n")

    result = GrepTool().call(
        call_id="call_1",
        pattern=r"interface\{\}",
        path=str(target),
        output_mode="files_with_matches",
    )

    assert len(result.output["matches"]) == 1


def test_grep_tool_reports_missing_path() -> None:
    """Missing paths surface as tool errors."""
    result = GrepTool().call(
        call_id="call_1",
        pattern="hello",
        path="/nonexistent/path/for/grep",
    )

    assert isinstance(result, ToolErrorMessage)


def test_grep_tool_reports_invalid_pattern() -> None:
    """Invalid regex patterns surface as tool errors."""
    result = GrepTool().call(
        call_id="call_1",
        pattern="[unclosed",
    )

    assert isinstance(result, ToolErrorMessage)
