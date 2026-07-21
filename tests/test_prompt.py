"""Tests for the prompt manager."""

from pathlib import Path

import pytest

from arancio.prompt.manager import PromptManager


def test_resolve_prompt_returns_command_arguments_for_slash_command() -> None:
    """A slash prompt resolves to the command's name and arguments."""
    action_kwargs = PromptManager.resolve_prompt("/hello-world Sam 1")

    assert action_kwargs == {
        "command_name": "hello-world",
        "command_args": ["Sam", "1"],
    }


def test_resolve_prompt_returns_prompt_argument_for_plain_prompt() -> None:
    """A prompt without a leading slash resolves to the raw prompt text."""
    action_kwargs = PromptManager.resolve_prompt("normal prompt")

    assert action_kwargs == {"prompt": "normal prompt", "mentions": []}


def test_resolve_prompt_returns_prompt_argument_for_empty_prompt() -> None:
    """An empty prompt resolves to the raw prompt text."""
    assert PromptManager.resolve_prompt("") == {"prompt": "", "mentions": []}


def test_resolve_prompt_returns_prompt_argument_for_lone_slash() -> None:
    """A lone slash carries no command name and resolves to the raw prompt text."""
    assert PromptManager.resolve_prompt("/") == {"prompt": "/", "mentions": []}


def test_resolve_prompt_rewrites_resolving_mention_to_absolute_path(
    tmp_path: Path,
) -> None:
    """A mention resolving to an existing file is rewritten to its absolute path."""
    target = tmp_path / "notes.md"
    target.write_text("note")

    action_kwargs = PromptManager.resolve_prompt(f"see @{target} please")

    assert action_kwargs == {
        "prompt": f"see @{target} please",
        "mentions": [target],
    }


def test_resolve_prompt_resolves_relative_mention_against_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative mention resolves against the process working directory."""
    target = tmp_path / "notes.md"
    target.write_text("note")
    monkeypatch.chdir(tmp_path)

    action_kwargs = PromptManager.resolve_prompt("see @notes.md please")

    assert action_kwargs == {
        "prompt": f"see @{target} please",
        "mentions": [target],
    }


def test_resolve_prompt_leaves_non_resolving_mention_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mention that does not resolve to a file is left unchanged."""
    monkeypatch.chdir(tmp_path)

    action_kwargs = PromptManager.resolve_prompt("reach me at user@example.com")

    assert action_kwargs == {
        "prompt": "reach me at user@example.com",
        "mentions": [],
    }


def test_resolve_prompt_directory_target_is_not_a_mention(tmp_path: Path) -> None:
    """A mention pointing at a directory, not a file, is left unchanged."""
    directory = tmp_path / "adir"
    directory.mkdir()

    prompt = f"see @{directory} please"
    action_kwargs = PromptManager.resolve_prompt(prompt)

    assert action_kwargs == {"prompt": prompt, "mentions": []}


def test_resolve_prompt_multiple_mentions_preserve_order(tmp_path: Path) -> None:
    """Multiple resolving mentions are rewritten and collected in order."""
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("first")
    second.write_text("second")

    action_kwargs = PromptManager.resolve_prompt(f"see @{first} and @{second}")

    assert action_kwargs == {
        "prompt": f"see @{first} and @{second}",
        "mentions": [first, second],
    }


def test_resolve_prompt_resolves_quoted_mention_with_spaces(tmp_path: Path) -> None:
    """A quoted mention can reference a path containing spaces."""
    target = tmp_path / "my notes.md"
    target.write_text("note")

    action_kwargs = PromptManager.resolve_prompt(f'see @"{target}" please')

    assert action_kwargs == {
        "prompt": f'see @"{target}" please',
        "mentions": [target],
    }


def test_resolve_prompt_rewrites_relative_mention_with_spaces_using_quotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resolving mention whose absolute path has a space is quoted when rewritten."""
    target = tmp_path / "my notes.md"
    target.write_text("note")
    monkeypatch.chdir(tmp_path)

    action_kwargs = PromptManager.resolve_prompt('see @"my notes.md" please')

    assert action_kwargs == {
        "prompt": f'see @"{target}" please',
        "mentions": [target],
    }


def test_resolve_prompt_command_arguments_are_not_mention_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """@mentions inside a slash command's arguments are left untouched."""
    target = tmp_path / "real.md"
    target.write_text("content")
    monkeypatch.chdir(tmp_path)

    action_kwargs = PromptManager.resolve_prompt(f"/cmd @{target}")

    assert action_kwargs == {
        "command_name": "cmd",
        "command_args": [f"@{target}"],
    }
