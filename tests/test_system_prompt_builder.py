"""Tests for the system prompt builder."""

from datetime import date
from pathlib import Path

import pytest
from dynamic_markdown.types.files.base import DynamicMarkdownFile

from arancio.core.builders import system_prompt as system_prompt_builder
from arancio.core.builders.system_prompt import SystemPromptBuilder
from arancio.core.constants.agent import AGENT_DEFAULT_SYSTEM_PROMPT
from arancio.storage import manager as storage_manager


def _patch_harness_path(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """Point the storage manager's harness path constant at ``path``."""
    monkeypatch.setattr(storage_manager, "SYSTEM_PROMPT_HARNESS_PATH", path)


def _with_context(
    prompt: str, today: date | None = None, cwd: Path | None = None
) -> str:
    """Return ``prompt`` with the runtime context suffix ``build`` appends."""
    today = today or date.today()
    cwd = cwd or Path.cwd()
    return f"{prompt}\n\nToday: {today.isoformat()}\nCurrent working directory: {cwd}"


def test_build_returns_repo_harness_system_prompt_content() -> None:
    """``build`` returns the parsed content of the repo's harness/SYSTEM_PROMPT.md."""
    expected_file = DynamicMarkdownFile(Path("harness/SYSTEM_PROMPT.md"))

    assert SystemPromptBuilder().build() == _with_context(expected_file.content)


def test_build_expands_dynamic_markdown_include_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A harness file with an ``<include>`` tag is expanded before being returned."""
    (tmp_path / "other.md").write_text("included")
    target = tmp_path / "SYSTEM_PROMPT.md"
    target.write_text("before <include>other.md</include> after")
    _patch_harness_path(monkeypatch, target)

    expected_file = DynamicMarkdownFile(target)
    assert SystemPromptBuilder().build() == _with_context(expected_file.content)


def test_build_writes_and_returns_fallback_when_harness_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing harness file is created with the fallback prompt and returned."""
    target = tmp_path / "nested" / "SYSTEM_PROMPT.md"
    _patch_harness_path(monkeypatch, target)

    assert SystemPromptBuilder().build() == _with_context(AGENT_DEFAULT_SYSTEM_PROMPT)
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

    assert builder.build() == builder.build() == _with_context("first")


def test_reload_refreshes_cached_prompt_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``reload()`` re-reads the harness file and replaces the cached prompt."""
    target = tmp_path / "SYSTEM_PROMPT.md"
    target.write_text("first")
    _patch_harness_path(monkeypatch, target)
    builder = SystemPromptBuilder()

    target.write_text("second")
    builder.reload()

    assert builder.build() == _with_context("second")


def test_reload_reseeds_default_when_harness_file_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``reload()`` recreates a deleted harness file instead of raising."""
    target = tmp_path / "SYSTEM_PROMPT.md"
    target.write_text("first")
    _patch_harness_path(monkeypatch, target)
    builder = SystemPromptBuilder()

    target.unlink()
    builder.reload()

    assert target.read_text() == AGENT_DEFAULT_SYSTEM_PROMPT
    assert builder.build() == _with_context(AGENT_DEFAULT_SYSTEM_PROMPT)


def test_build_appends_current_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``build()`` appends today's date to the cached prompt."""
    target = tmp_path / "SYSTEM_PROMPT.md"
    target.write_text("prompt body")
    _patch_harness_path(monkeypatch, target)

    assert SystemPromptBuilder().build() == (
        f"prompt body\n\nToday: {date.today().isoformat()}"
        f"\nCurrent working directory: {Path.cwd()}"
    )


def test_build_appends_current_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``build()`` appends the working directory to the cached prompt."""
    target = tmp_path / "SYSTEM_PROMPT.md"
    target.write_text("prompt body")
    _patch_harness_path(monkeypatch, target)
    monkeypatch.chdir(tmp_path)

    assert SystemPromptBuilder().build() == _with_context("prompt body", cwd=Path.cwd())


def test_build_tracks_working_directory_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The working directory is resolved per call, so a move is picked up.

    ``App.set_working_directory`` mirrors every move to ``os.chdir``, so this stands in
    for a ``/cd`` performed mid-session.
    """
    target = tmp_path / "SYSTEM_PROMPT.md"
    target.write_text("prompt body")
    _patch_harness_path(monkeypatch, target)
    moved_to = tmp_path / "sub"
    moved_to.mkdir()

    monkeypatch.chdir(tmp_path)
    builder = SystemPromptBuilder()
    before = builder.build()
    monkeypatch.chdir(moved_to)
    after = builder.build()

    assert before != after
    assert before.endswith(f"Current working directory: {tmp_path.resolve()}")
    assert after.endswith(f"Current working directory: {moved_to.resolve()}")


def test_build_refreshes_date_on_every_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The date is resolved per call, so it tracks a session crossing midnight."""
    target = tmp_path / "SYSTEM_PROMPT.md"
    target.write_text("prompt body")
    _patch_harness_path(monkeypatch, target)
    builder = SystemPromptBuilder()

    class _StubDate(date):
        _today = date(2026, 7, 30)

        @classmethod
        def today(cls) -> date:
            return cls._today

    monkeypatch.setattr(system_prompt_builder, "date", _StubDate)
    first = builder.build()
    _StubDate._today = date(2026, 7, 31)
    second = builder.build()

    assert first == _with_context("prompt body", date(2026, 7, 30))
    assert second == _with_context("prompt body", date(2026, 7, 31))
