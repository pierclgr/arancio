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


def test_resolve_prompt_returns_hidden_shell_command_verbatim() -> None:
    """A leading ``!!`` makes the entire remainder a hidden shell command."""
    command = 'printf "%s\\n" "@notes.md" | sed "s/a/b/"'

    action_kwargs = PromptManager.resolve_prompt(f"!!{command}")

    assert action_kwargs == {
        "shell_command": command,
        "add_to_history": False,
    }


def test_resolve_prompt_hidden_shell_command_stops_further_parsing() -> None:
    """Slash commands and mentions inside ``!!`` remain shell syntax."""
    command = "/model gpt-5 @notes.md"

    action_kwargs = PromptManager.resolve_prompt(f"!!{command}")

    assert action_kwargs == {
        "shell_command": command,
        "add_to_history": False,
    }


def test_resolve_prompt_rejects_empty_hidden_shell_command() -> None:
    """A bare ``!!`` is invalid because it contains no shell command."""
    with pytest.raises(ValueError, match="shell command is empty"):
        PromptManager.resolve_prompt("!!")


def test_resolve_prompt_keeps_a_quoted_argument_with_spaces_together() -> None:
    """A double-quoted argument stays one word and loses its surrounding quotes."""
    action_kwargs = PromptManager.resolve_prompt('/cd "test/of path/"')

    assert action_kwargs == {
        "command_name": "cd",
        "command_args": ["test/of path/"],
    }


def test_resolve_prompt_keeps_a_single_quoted_argument_with_spaces_together() -> None:
    """A single-quoted argument is handled like a double-quoted one."""
    action_kwargs = PromptManager.resolve_prompt("/cd 'my folder'")

    assert action_kwargs == {"command_name": "cd", "command_args": ["my folder"]}


def test_resolve_prompt_splits_quoted_and_bare_arguments_together() -> None:
    """Quoted and bare arguments mix, each arriving as its own word."""
    action_kwargs = PromptManager.resolve_prompt('/hello-world "Sam Smith" 2')

    assert action_kwargs == {
        "command_name": "hello-world",
        "command_args": ["Sam Smith", "2"],
    }


def test_resolve_prompt_keeps_an_apostrophe_inside_a_word_literal() -> None:
    """A quote that does not start a word is literal and needs no closing."""
    action_kwargs = PromptManager.resolve_prompt("/hello-world O'Brien")

    assert action_kwargs == {"command_name": "hello-world", "command_args": ["O'Brien"]}


def test_resolve_prompt_preserves_backslashes_in_command_arguments() -> None:
    """Backslashes stay literal, so a Windows path survives splitting."""
    action_kwargs = PromptManager.resolve_prompt(r"/cd C:\Users\me")

    assert action_kwargs == {"command_name": "cd", "command_args": [r"C:\Users\me"]}


def test_resolve_prompt_rejects_an_unclosed_quote() -> None:
    """A quote opening a word but never closed raises rather than mis-splitting."""
    with pytest.raises(ValueError, match="unbalanced quote"):
        PromptManager.resolve_prompt('/cd "my folder')


def test_resolve_prompt_returns_empty_arguments_for_a_bare_command() -> None:
    """A command with no argument text resolves to an empty argument list."""
    assert PromptManager.resolve_prompt("/clear") == {
        "command_name": "clear",
        "command_args": [],
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


def test_resolve_prompt_directory_target_resolves_as_a_mention(tmp_path: Path) -> None:
    """A mention pointing at an existing directory resolves like a file mention."""
    directory = tmp_path / "adir"
    directory.mkdir()

    prompt = f"see @{directory} please"
    action_kwargs = PromptManager.resolve_prompt(prompt)

    assert action_kwargs == {
        "prompt": f"see @{directory} please",
        "mentions": [directory],
    }


def test_resolve_prompt_missing_path_is_not_a_mention(tmp_path: Path) -> None:
    """A mention pointing at neither a file nor a directory is left unchanged."""
    missing = tmp_path / "missing"

    prompt = f"see @{missing} please"
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
