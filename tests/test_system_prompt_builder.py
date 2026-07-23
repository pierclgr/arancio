"""Tests for the system prompt builder."""

from pathlib import Path

import pytest
from dynamic_markdown.types.files.base import DynamicMarkdownFile

from arancio.core.builders import system_prompt as system_prompt_builder
from arancio.core.builders.system_prompt import SystemPromptBuilder


@pytest.fixture(autouse=True)
def _reset_system_prompt_cache() -> None:
    """Clear the cached system prompt before each test for isolation."""
    SystemPromptBuilder._prompt = None


def test_build_returns_repo_harness_system_prompt_content() -> None:
    """``build`` returns the parsed content of the repo's harness/SYSTEM_PROMPT.md."""
    expected_file = DynamicMarkdownFile(Path("harness/SYSTEM_PROMPT.md"))
    expected_file.parse()

    assert SystemPromptBuilder.build() == expected_file.content


def test_build_expands_dynamic_markdown_include_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A harness file with an ``<include>`` tag is expanded before being returned."""
    (tmp_path / "other.md").write_text("included")
    target = tmp_path / "SYSTEM_PROMPT.md"
    target.write_text("before <include>other.md</include> after")
    monkeypatch.setattr(system_prompt_builder, "SYSTEM_PROMPT_HARNESS_PATH", target)

    expected_file = DynamicMarkdownFile(target)
    expected_file.parse()
    assert SystemPromptBuilder.build() == expected_file.content


def test_build_raises_when_harness_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing harness file raises ``FileNotFoundError``."""
    target = tmp_path / "SYSTEM_PROMPT.md"
    monkeypatch.setattr(system_prompt_builder, "SYSTEM_PROMPT_HARNESS_PATH", target)

    with pytest.raises(FileNotFoundError, match="SYSTEM_PROMPT.md"):
        SystemPromptBuilder.build()


def test_build_caches_prompt_after_first_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second ``build()`` call returns the cached prompt, not a fresh reread."""
    target = tmp_path / "SYSTEM_PROMPT.md"
    target.write_text("first")
    monkeypatch.setattr(system_prompt_builder, "SYSTEM_PROMPT_HARNESS_PATH", target)

    first = SystemPromptBuilder.build()
    target.write_text("second")
    second = SystemPromptBuilder.build()

    assert first == second == "first"
