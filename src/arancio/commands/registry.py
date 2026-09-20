"""Registry of available slash commands."""

from typing import Dict, FrozenSet, Type

from arancio.commands.base import BaseCommand
from arancio.commands.cd import CdCommand
from arancio.commands.clear import ClearCommand
from arancio.commands.effort import EffortCommand
from arancio.commands.exit import ExitCommand
from arancio.commands.fork import ForkCommand
from arancio.commands.hello_world import HelloWorldCommand
from arancio.commands.model import ModelCommand
from arancio.commands.permissions import PermissionsCommand
from arancio.commands.provider import ProviderCommand
from arancio.commands.rename import RenameCommand
from arancio.commands.resume import ResumeCommand

# register a command by adding an entry keyed by its name; aliases point at the
# same class under an extra key
COMMAND_REGISTRY: Dict[str, Type[BaseCommand]] = {
    HelloWorldCommand.name: HelloWorldCommand,
    ExitCommand.name: ExitCommand,
    "quit": ExitCommand,
    ModelCommand.name: ModelCommand,
    EffortCommand.name: EffortCommand,
    PermissionsCommand.name: PermissionsCommand,
    ProviderCommand.name: ProviderCommand,
    ClearCommand.name: ClearCommand,
    "new": ClearCommand,
    CdCommand.name: CdCommand,
    ResumeCommand.name: ResumeCommand,
    RenameCommand.name: RenameCommand,
    ForkCommand.name: ForkCommand,
}

# a command that ends or switches the active chat leaves nothing behind in the one
# it was typed into, which decides both when the executor records its typed line
# (before it runs, since the chat it belongs to is about to be replaced) and
# whether anything is written at all (nothing is, while that chat has no log yet).
# keyed by class, so the "quit" and "new" aliases are covered too
SESSION_DISCARDING_COMMANDS: FrozenSet[Type[BaseCommand]] = frozenset(
    {ExitCommand, ClearCommand, ForkCommand, ResumeCommand}
)
