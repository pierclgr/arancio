"""Base class every plugin extends."""

from abc import ABC
from pathlib import Path

from arancio.core.plugins.manifest import PluginManifest


class Plugin(ABC):
    """Base class every plugin extends.

    A plugin runs on hooks via ``@hook``-decorated methods (see
    :mod:`~arancio.core.plugins.hook`), defines tools via ``@tool``-decorated
    methods (see :mod:`~arancio.core.plugins.tool`), or both. A plugin
    reaches the extra files shipped in its own folder through
    :attr:`directory`; the loader never reads them.

    Attributes:
        manifest: the metadata parsed from the plugin's ``manifest.yml``.
        directory: the plugin's own folder.
    """

    def __init__(self, manifest: PluginManifest, directory: Path) -> None:
        """Initialize the plugin with its metadata and its own folder.

        Args:
            manifest: the metadata parsed from the plugin's ``manifest.yml``.
            directory: the plugin's own folder, holding any extra files it
                ships.
        """
        self.manifest: PluginManifest = manifest
        self.directory: Path = directory

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
