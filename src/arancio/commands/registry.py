"""Registry of available slash commands."""

from typing import Dict, Type

from arancio.commands.base import BaseCommand
from arancio.commands.cd import CdCommand
from arancio.commands.clear import ClearCommand
from arancio.commands.effort import EffortCommand
from arancio.commands.exit import ExitCommand
from arancio.commands.hello_world import HelloWorldCommand
from arancio.commands.model import ModelCommand
from arancio.commands.permissions import PermissionsCommand
from arancio.commands.provider import ProviderCommand

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
}
