"""Base class every plugin extends."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from arancio.core.hooks.types import Hook
from arancio.core.plugins.manifest import PluginManifest


class Plugin(ABC):
    """Base class every plugin extends.

    A plugin binds to one or more hook entry points via :attr:`hooks` and
    implements :meth:`execute` to run on each one. A plugin reaches the extra
    files shipped in its own folder through :attr:`directory`; the loader
    never reads them.

    Attributes:
        manifest: the metadata parsed from the plugin's ``manifest.yml``.
        directory: the plugin's own folder.
        hooks: the hooks this plugin runs on.
    """

    hooks: ClassVar[frozenset[Hook]] = frozenset()

    def __init__(self, manifest: PluginManifest, directory: Path) -> None:
        """Initialize the plugin with its metadata and its own folder.

        Args:
            manifest: the metadata parsed from the plugin's ``manifest.yml``.
            directory: the plugin's own folder, holding any extra files it
                ships.
        """
        self.manifest = manifest
        self.directory = directory

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the plugin.

        Returns:
            The plugin's class name followed by its folder name.
        """
        return f"{type(self).__name__}({self.directory.name})"

    @property
    def name(self) -> str:
        """Return the plugin's display name.

        Returns:
            The manifest's name, which defaults to the plugin's folder name.
        """
        return self.manifest.name

    @abstractmethod
    def execute(self, *, hook: Hook, **kwargs: Any) -> None:
        """Run the plugin for one dispatched hook.

        Args:
            hook: the hook that fired.
                :meth:`~arancio.core.hooks.manager.HookManager.run` forwards
                only the hook's own keyword arguments and never says which
                hook it is dispatching, so the plugin manager passes it here —
                a plugin bound to two hooks could not otherwise tell them
                apart.
            **kwargs: that hook's own keyword arguments, unchanged. See
                AGENTS.md's Hooks section for each hook's contract.

        Raises:
            NotImplementedError: when the method is not implemented by the
                subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")
