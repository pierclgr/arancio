"""Action types produced by the prompt manager."""

from abc import ABC
from dataclasses import dataclass


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
class PromptAction(BaseAction):
    """Send the raw prompt text to the model.

    Attributes:
        prompt: the raw prompt to send to the model.
    """

    prompt: str
