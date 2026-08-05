"""Action types produced by the prompt manager."""

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path


class BaseAction(ABC):
    """Base class for a resolved decision about how to handle a prompt.

    :class:`arancio.prompt.manager.PromptManager` turns a raw prompt into a
    concrete action, and :class:`arancio.prompt.actions.executor.ActionExecutor`
    carries it out. Actions are plain data: they hold what the executor needs and
    do not execute themselves.
    """


@dataclass
class CommandAction(BaseAction):
    """Run the named slash command with its parsed arguments.

    Attributes:
        name: the command name parsed from the prompt.
        args: the command arguments parsed from the prompt.
    """

    name: str
    args: list[str]


@dataclass
class ShellCommandAction(BaseAction):
    """Run user-provided text as a shell command.

    Attributes:
        command: the exact command text following the shell prefix.
        add_to_history: whether to store the generated tool call and result
            in the agent's conversation history.
    """

    command: str
    add_to_history: bool


@dataclass
class PromptAction(BaseAction):
    """Send the raw prompt text to the model.

    Attributes:
        prompt: the prompt to send to the model, with every resolving
            @mention already rewritten to its resolved absolute path.
        mentions: absolute paths resolved from @mention tokens in the
            prompt, in the order they appear. Empty when the prompt had
            none.
    """

    prompt: str
    mentions: list[Path] = field(default_factory=list)
