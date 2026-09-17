"""Tests for how a typed line is classified into an action.

``resolve_prompt`` is the first thing that touches user input, and its ordering is load-
bearing: a shell prefix wins over everything, so a ``/`` or an ``@`` inside a shell
command is left for the shell to interpret, not for us.
"""

from pathlib import Path

import pytest

from arancio.prompt.actions.factory import ActionFactory
from arancio.prompt.actions.types import (
    CommandAction,
    PromptAction,
    ShellCommandAction,
)
from arancio.prompt.manager import PromptManager


def test_one_bang_runs_a_command_the_model_will_see() -> None:
    """``!`` is "run this and remember it", so the pair joins model history."""
    resolved = PromptManager.resolve_prompt("!ls -la")

    assert resolved == {"shell_command": "ls -la", "add_to_history": True}


def test_two_bangs_run_a_command_the_model_will_not_see() -> None:
    """``!!`` is "run this for me alone"."""
    resolved = PromptManager.resolve_prompt("!!ls -la")

    assert resolved == {"shell_command": "ls -la", "add_to_history": False}


def test_a_shell_command_is_handed_over_completely_unparsed() -> None:
    """Slashes, ``@`` and newlines all belong to the shell once ``!`` matched."""
    resolved = PromptManager.resolve_prompt("! grep /etc @file\nsecond line")

    assert resolved["shell_command"] == " grep /etc @file\nsecond line"


def test_a_shell_prefix_with_nothing_after_it_is_refused() -> None:
    """Running an empty command would be a silent no-op the user cannot see."""
    with pytest.raises(ValueError):
        PromptManager.resolve_prompt("!   ")


def test_a_slash_command_splits_into_a_name_and_words() -> None:
    """The words become the command's positional arguments, in order."""
    resolved = PromptManager.resolve_prompt("/permissions read auto")

    assert resolved == {
        "command_name": "permissions",
        "command_args": ["read", "auto"],
    }


def test_a_slash_command_without_arguments_gets_an_empty_list() -> None:
    """``/permissions`` alone is the read form, so no arguments is valid."""
    resolved = PromptManager.resolve_prompt("/permissions")

    assert resolved == {"command_name": "permissions", "command_args": []}


def test_a_quoted_argument_survives_as_one_word() -> None:
    """A path with a space is one argument, not two."""
    resolved = PromptManager.resolve_prompt('/cd "my folder"')

    assert resolved["command_args"] == ["my folder"]


def test_an_apostrophe_inside_a_word_needs_no_closing_quote() -> None:
    """Otherwise ordinary English would read as an unbalanced quote."""
    resolved = PromptManager.resolve_prompt("/cd don't")

    assert resolved["command_args"] == ["don't"]


def test_an_unclosed_quote_is_reported_against_the_arguments() -> None:
    """The error names the argument text, rather than failing further down."""
    with pytest.raises(ValueError, match="unbalanced quote"):
        PromptManager.resolve_prompt('/cd "my folder')


def test_a_lone_slash_is_just_text() -> None:
    """A command needs a name, so a bare slash goes to the model."""
    resolved = PromptManager.resolve_prompt("/")

    assert resolved == {"prompt": "/", "mentions": []}


def test_plain_text_becomes_a_prompt_with_no_mentions() -> None:
    """The common case carries nothing extra."""
    resolved = PromptManager.resolve_prompt("what does this do?")

    assert resolved == {"prompt": "what does this do?", "mentions": []}


def test_a_mention_that_exists_is_rewritten_to_its_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model gets a path it can act on, not whatever the user shortened."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.md").write_text("x")

    resolved = PromptManager.resolve_prompt("look at @notes.md please")

    target = (tmp_path / "notes.md").resolve()
    assert resolved["mentions"] == [target]
    assert resolved["prompt"] == f"look at @{target} please"


def test_a_mention_that_does_not_exist_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An email address or a handle is not a file, and must not be mangled."""
    monkeypatch.chdir(tmp_path)

    resolved = PromptManager.resolve_prompt("ask @someone about it")

    assert resolved == {"prompt": "ask @someone about it", "mentions": []}


def test_a_mentioned_path_containing_a_space_is_requoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the quotes the rewritten path would split at the space."""
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "my notes"
    folder.mkdir()

    resolved = PromptManager.resolve_prompt('read @"my notes"')

    assert resolved["prompt"] == f'read @"{folder.resolve()}"'
    assert resolved["mentions"] == [folder.resolve()]


def test_a_directory_mention_resolves_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory is listed rather than read, but it still resolves."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()

    resolved = PromptManager.resolve_prompt("check @src")

    assert resolved["mentions"] == [(tmp_path / "src").resolve()]


def test_the_factory_builds_the_action_the_keywords_describe() -> None:
    """The factory is the only place that knows which keys mean which action."""
    shell = ActionFactory.create_action(
        raw_input="!ls", shell_command="ls", add_to_history=True
    )
    command = ActionFactory.create_action(
        raw_input="/cd /tmp", command_name="cd", command_args=["/tmp"]
    )
    prompt = ActionFactory.create_action(raw_input="hi", prompt="hi", mentions=[])

    assert isinstance(shell, ShellCommandAction)
    assert isinstance(command, CommandAction)
    assert isinstance(prompt, PromptAction)


def test_every_action_carries_the_line_the_user_typed() -> None:
    """Nothing downstream has to approximate it: the app always supplies it."""
    action = ActionFactory.create_action(
        raw_input="/model gpt-5", command_name="model", command_args=["gpt-5"]
    )

    assert action.raw_input == "/model gpt-5"


def test_an_action_cannot_be_built_without_the_typed_line() -> None:
    """It is required, so an approximation can never silently take its place."""
    with pytest.raises(TypeError):
        PromptAction(prompt="hi")
