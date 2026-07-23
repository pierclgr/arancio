"""Tests for the system prompt builder."""

from pathlib import Path

import pytest
from dynamic_markdown.types.files.base import DynamicMarkdownFile

from arancio.core.builders.system_prompt import SystemPromptBuilder
from arancio.core.constants.agent import AGENT_DEFAULT_SYSTEM_PROMPT
from arancio.storage import manager as storage_manager


def _patch_harness_path(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """Point the storage manager's harness path constant at ``path``."""
    monkeypatch.setattr(storage_manager, "SYSTEM_PROMPT_HARNESS_PATH", path)


def test_build_returns_repo_harness_system_prompt_content() -> None:
    """``build`` returns the parsed content of the repo's harness/SYSTEM_PROMPT.md."""
    expected_file = DynamicMarkdownFile(Path("harness/SYSTEM_PROMPT.md"))

    assert SystemPromptBuilder().build() == expected_file.content


def test_build_expands_dynamic_markdown_include_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A harness file with an ``<include>`` tag is expanded before being returned."""
    (tmp_path / "other.md").write_text("included")
    target = tmp_path / "SYSTEM_PROMPT.md"
    target.write_text("before <include>other.md</include> after")
    _patch_harness_path(monkeypatch, target)

    expected_file = DynamicMarkdownFile(target)
    assert SystemPromptBuilder().build() == expected_file.content


def test_build_writes_and_returns_fallback_when_harness_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing harness file is created with the fallback prompt and returned."""
    target = tmp_path / "nested" / "SYSTEM_PROMPT.md"
    _patch_harness_path(monkeypatch, target)

    assert SystemPromptBuilder().build() == AGENT_DEFAULT_SYSTEM_PROMPT
    assert target.read_text() == AGENT_DEFAULT_SYSTEM_PROMPT


def test_build_returns_prompt_cached_at_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``build()`` returns the prompt loaded at construction, not a fresh reread."""
    target = tmp_path / "SYSTEM_PROMPT.md"
    target.write_text("first")
    _patch_harness_path(monkeypatch, target)

    builder = SystemPromptBuilder()
    target.write_text("second")

    assert builder.build() == builder.build() == "first"
