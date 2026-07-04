"""Tests for the prompt manager."""

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

    assert action_kwargs == {"prompt": "normal prompt"}


def test_resolve_prompt_returns_prompt_argument_for_empty_prompt() -> None:
    """An empty prompt resolves to the raw prompt text."""
    assert PromptManager.resolve_prompt("") == {"prompt": ""}


def test_resolve_prompt_returns_prompt_argument_for_lone_slash() -> None:
    """A lone slash carries no command name and resolves to the raw prompt text."""
    assert PromptManager.resolve_prompt("/") == {"prompt": "/"}
