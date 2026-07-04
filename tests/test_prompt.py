"""Tests for the prompt manager."""

from arancio.prompt.actions.types import CommandAction, PromptAction
from arancio.prompt.manager import PromptManager


def test_resolve_prompt_returns_command_action_for_slash_command() -> None:
    """A slash prompt resolves to a command action with its name and arguments."""
    action = PromptManager.resolve_prompt("/hello-world Sam 1")

    assert action == CommandAction(name="hello-world", args=["Sam", "1"])


def test_resolve_prompt_returns_prompt_action_for_plain_prompt() -> None:
    """A prompt without a leading slash resolves to a prompt action."""
    action = PromptManager.resolve_prompt("normal prompt")

    assert action == PromptAction(prompt="normal prompt")


def test_resolve_prompt_returns_prompt_action_for_empty_prompt() -> None:
    """An empty prompt resolves to a prompt action."""
    assert PromptManager.resolve_prompt("") == PromptAction(prompt="")


def test_resolve_prompt_returns_prompt_action_for_lone_slash() -> None:
    """A lone slash carries no command name and resolves to a prompt action."""
    assert PromptManager.resolve_prompt("/") == PromptAction(prompt="/")
