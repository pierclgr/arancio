"""Tests for the App's working directory management."""

import os
from pathlib import Path

import pytest

from arancio.settings.settings import Settings
from arancio.ui.app import App


class _DummyAgent:
    """Stand-in agent; the working directory tests never run its loop."""


class _FakeSettingsManager:
    """Settings-manager stub without disk I/O."""

    def __init__(self) -> None:
        """Build default settings."""
        from arancio.core.permissions.types import PermissionCategory, PermissionLevel

        self.settings = Settings(
            permissions={c: PermissionLevel.ASK for c in PermissionCategory},
            provider="openai",
            model_name="gpt-4o",
            thinking_effort="medium",
            thinking_summary=None,
            max_turns="inf",
            max_retries=3,
            turn_wait_time=1.0,
            turn_wait_time_multiplier=2.0,
        )


def _chdir_safe(path: Path) -> None:
    """``os.chdir`` guarded so a missing path never fails teardown."""
    if path.is_dir():
        os.chdir(path)


@pytest.fixture(autouse=True)
def _restore_cwd() -> None:
    """Restore the process cwd after each test.

    Tests call ``App.set_working_directory``, which mirrors to ``os.chdir``;
    this captures the original cwd once and resets it on teardown.

    Yields:
        ``None``; the teardown runs after the test body.
    """
    original = Path.cwd()
    yield
    _chdir_safe(original)


def _app() -> App:
    """Build an app wired to a dummy agent.

    Returns:
        An :class:`App` instance with a dummy agent.
    """
    return App(
        agent=_DummyAgent(),
        model_id="test-model",
        settings_manager=_FakeSettingsManager(),
    )


def test_construction_captures_current_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The app captures the process cwd at construction time."""
    monkeypatch.chdir(tmp_path)

    app = _app()

    assert app.working_directory == tmp_path


def test_working_directory_property_returns_the_held_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``working_directory`` exposes the held directory unchanged."""
    monkeypatch.chdir(tmp_path)
    app = _app()

    assert app.working_directory == tmp_path


def test_set_working_directory_updates_app_and_mirrors_to_process_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``set_working_directory`` moves the app and the process cwd together."""
    target = tmp_path / "sub"
    target.mkdir()
    monkeypatch.chdir(tmp_path)
    app = _app()

    app.set_working_directory(target)

    assert app.working_directory == target
    assert Path.cwd() == target


def test_set_working_directory_resolves_relative_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative path is resolved against the current process cwd."""
    target = tmp_path / "rel"
    target.mkdir()
    monkeypatch.chdir(tmp_path)
    app = _app()

    app.set_working_directory("rel")

    assert app.working_directory == target


def test_set_working_directory_expands_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``~``-prefixed path is expanded before resolution."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("USER", "test")
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)
    app = _app()

    app.set_working_directory("~")

    assert app.working_directory == tmp_path
    assert Path.cwd() == tmp_path


def test_set_working_directory_rejects_nonexistent_path(tmp_path: Path) -> None:
    """A path that does not exist reports that it does not exist."""
    app = _app()
    missing = tmp_path / "nope"

    with pytest.raises(ValueError, match="does not exist"):
        app.set_working_directory(missing)


def test_set_working_directory_rejects_file_path(tmp_path: Path) -> None:
    """A path pointing at a file (not a directory) raises ``ValueError``."""
    file = tmp_path / "afile"
    file.write_text("x")
    app = _app()

    with pytest.raises(ValueError, match="not a directory"):
        app.set_working_directory(file)
