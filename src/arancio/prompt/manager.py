"""Prompt manager turning a raw prompt into an action."""

from arancio.prompt.actions.factory import ActionFactory
from arancio.prompt.actions.types import BaseAction
from arancio.prompt.patterns import COMMAND_PATTERN


class PromptManager:
    """Resolves a raw prompt into the action that should handle it.

    A prompt starting with ``/`` is parsed as ``/<name> <args...>``: the first word is
    the command name and the remaining whitespace-separated words are its arguments,
    producing a :class:`arancio.prompt.actions.types.CommandAction`. Any other
    prompt produces a :class:`arancio.prompt.actions.types.PromptAction` carrying
    the text for the model. The action is handed to
    :class:`arancio.prompt.actions.executor.ActionExecutor` to be carried out.
    """

    @classmethod
    def resolve_prompt(cls, prompt: str) -> BaseAction:
        """Resolve the prompt into the action that should handle it.

        Args:
            prompt: the raw user prompt.

        Returns:
            A :class:`CommandAction` when the prompt is a slash command, otherwise
            a :class:`PromptAction` carrying the prompt text for the model.
        """
        match = COMMAND_PATTERN.match(prompt)
        if match is None:
            return ActionFactory.create_prompt_action(prompt)

        command_name = match.group(1)
        command_args = match.group(2).split() if match.group(2) else []
        return ActionFactory.create_command_action(command_name, command_args)
