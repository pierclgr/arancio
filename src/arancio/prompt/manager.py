"""Prompt manager turning a raw prompt into action arguments."""

from typing import Any

from arancio.prompt.patterns import COMMAND_PATTERN


class PromptManager:
    """Resolves a raw prompt into the keyword arguments for its action.

    A prompt starting with ``/`` is parsed as ``/<name> <args...>``: the first word is
    the command name and the remaining whitespace-separated words are its arguments,
    producing the ``command_name``/``command_args`` keyword arguments for a
    :class:`arancio.prompt.actions.types.CommandAction`. Any other prompt produces
    the ``prompt`` keyword argument for a
    :class:`arancio.prompt.actions.types.PromptAction`.
    :class:`arancio.prompt.actions.executor.ActionExecutor` builds and runs the
    action from these arguments.
    """

    @classmethod
    def resolve_prompt(cls, prompt: str) -> dict[str, Any]:
        """Resolve the prompt into the keyword arguments for its action.

        Args:
            prompt: the raw user prompt.

        Returns:
            The ``command_name``/``command_args`` keyword arguments for a command
            action when the prompt is a slash command, otherwise the ``prompt``
            keyword argument for a prompt action.
        """
        match = COMMAND_PATTERN.match(prompt)
        if match is None:
            return {"prompt": prompt}

        command_name = match.group(1)
        command_args = match.group(2).split() if match.group(2) else []
        return {"command_name": command_name, "command_args": command_args}
