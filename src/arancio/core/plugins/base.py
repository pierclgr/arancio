"""Plugin declarations and their shared runtime instances."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from arancio.core.hooks.base import Hook

if TYPE_CHECKING:
    from arancio.core.plugins.tools import PluginTool


@dataclass
class Plugin:
    """Keep a plugin's metadata and declarations outside its class.

    Attributes:
        plugin_cls: the original decorated class, constructed at startup.
        source_path: the file declaring the class.
        name: the plugin name with spaces replaced by dashes, defaulted to the
            class name.
        version: the declared version.
        description: the plugin's purpose.
        author: the declared plugin author.
        instance: the shared runtime instance, or None when inactive.
        hooks: methods and their hook events, bound once during loading.
        tools: tool classes holding metadata and methods bound during loading.
    """

    plugin_cls: type
    source_path: Path
    name: str
    version: str = "0.0.0"
    description: str = ""
    author: str = ""
    instance: object | None = field(default=None, init=False)
    hooks: list[tuple[Callable, tuple[Hook, ...]]] = field(
        default_factory=list, init=False
    )
    tools: list[type["PluginTool"]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Reject invalid metadata before registering a declaration.

        Raises:
            TypeError: if a metadata field has the wrong type.
            ValueError: if the plugin name is empty.
        """
        for name in ("name", "version", "description", "author"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be str, got {value!r}.")
        if not self.name:
            raise ValueError(f"name must not be empty, got {self.name!r}.")
        self.name = self.name.replace(" ", "-")

    @property
    def location(self) -> str:
        """The source location used in startup diagnostics.

        Returns:
            The plugin's relative file path and class name.
        """
        return (
            f"plugins/{self.source_path.parent.name}/{self.source_path.name}:"
            f"{self.plugin_cls.__name__}"
        )
