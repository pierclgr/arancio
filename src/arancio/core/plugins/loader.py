"""Import of one plugin folder into a live plugin instance."""

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

import yaml

from arancio.core.constants.path import (
    PLUGIN_MANIFEST_FILENAME,
    PLUGIN_MODULE_FILENAME,
)
from arancio.core.messages import ErrorMessage, Message
from arancio.core.plugins.base import Plugin
from arancio.core.plugins.constants import PLUGIN_PACKAGE_ROOT
from arancio.core.plugins.hook import find_hook_methods
from arancio.core.plugins.manifest import PluginManifest, PluginManifestValidator
from arancio.core.plugins.tool import find_tool_specs


class PluginLoader:
    """Turn one plugin folder into a plugin instance, or into an error.

    :meth:`load` never raises: every way a folder can be a broken plugin — unreadable or
    malformed ``manifest.yml``, a ``module.py`` that is missing or raises on import, a
    module holding no plugin class or several, a
    :class:`~arancio.core.plugins.base.Plugin` declaring neither hooks nor tools — comes
    back as one :class:`~arancio.core.messages.ErrorMessage` with no plugin, so a single
    bad folder never stops the others from loading.
    """

    @classmethod
    def load(cls, directory: Path) -> tuple[Plugin | None, list[Message]]:
        """Load one plugin folder.

        Args:
            directory: the plugin's folder, holding ``manifest.yml`` and
                ``module.py``.

        Returns:
            A ``(plugin, messages)`` pair. The plugin is ``None`` when the
            folder could not be loaded or the manifest disabled it; the
            messages describe anything that went wrong, and are empty for a
            clean load or a deliberately disabled plugin.
        """
        location = f"plugins/{directory.name}/{PLUGIN_MANIFEST_FILENAME}"
        manifest, messages = cls._read_manifest(directory, location)
        if manifest is None:
            return None, messages
        if not manifest.enabled:
            return None, messages

        module, errors = cls._import_module(directory)
        if module is None:
            return None, [*messages, *errors]

        plugin_cls, errors = cls._find_plugin_class(module, directory)
        if plugin_cls is None:
            return None, [*messages, *errors]

        if not find_hook_methods(plugin_cls) and not find_tool_specs(plugin_cls):
            return None, [
                *messages,
                ErrorMessage(
                    content=(
                        f"plugins/{directory.name}: {plugin_cls.__name__} declares "
                        "no hooks and no tools; plugin ignored."
                    )
                ),
            ]

        return plugin_cls(manifest=manifest, directory=directory), messages

    @staticmethod
    def _read_manifest(
        directory: Path, location: str
    ) -> tuple[PluginManifest | None, list[Message]]:
        """Read and validate the folder's manifest file.

        Args:
            directory: the plugin's folder.
            location: the manifest's path as it should read in messages.

        Returns:
            A ``(manifest, messages)`` pair; the manifest is ``None`` when the
            file could not be read or does not hold a mapping.
        """
        path = directory / PLUGIN_MANIFEST_FILENAME
        try:
            raw = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            return None, [ErrorMessage(content=f"{location}: cannot be read: {exc}")]

        # an empty manifest is valid: every field has a default
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            return None, [
                ErrorMessage(
                    content=f"{location}: must be a mapping, got {raw!r}; "
                    "plugin ignored."
                )
            ]

        return PluginManifestValidator.validate(raw, directory.name, location)

    @classmethod
    def _import_module(cls, directory: Path) -> tuple[ModuleType | None, list[Message]]:
        """Import the folder's ``module.py`` under its own synthetic package.

        The module is imported as ``arancio_plugins.<folder>.module``, with the
        parent package's search path pointing at the plugin folder, so
        ``from .helpers import X`` works inside ``module.py`` and two plugins
        each shipping a ``helpers.py`` never collide. ``sys.path`` is left
        untouched.

        Args:
            directory: the plugin's folder.

        Returns:
            A ``(module, messages)`` pair; the module is ``None`` when the
            import failed, and the messages say why.
        """
        path = directory / PLUGIN_MODULE_FILENAME
        if not path.is_file():
            return None, [
                ErrorMessage(
                    content=(
                        f"plugins/{directory.name}: {PLUGIN_MODULE_FILENAME} not "
                        "found; plugin ignored."
                    )
                )
            ]

        package = cls._register_package(directory)
        name = f"{package}.module"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            return None, [
                ErrorMessage(
                    content=(
                        f"plugins/{directory.name}/{PLUGIN_MODULE_FILENAME}: "
                        "cannot be imported as a module."
                    )
                )
            ]

        try:
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(name, None)
            return None, [
                ErrorMessage(
                    content=(
                        f"plugins/{directory.name}/{PLUGIN_MODULE_FILENAME}: "
                        f"failed to import: {exc}"
                    )
                )
            ]
        return module, []

    @staticmethod
    def _register_package(directory: Path) -> str:
        """Register the synthetic package a plugin's modules live under.

        Args:
            directory: the plugin's folder, which becomes the package's search
                path.

        Returns:
            The dotted name of the plugin's own package.
        """
        if PLUGIN_PACKAGE_ROOT not in sys.modules:
            root = ModuleType(PLUGIN_PACKAGE_ROOT)
            root.__path__ = []  # type: ignore[attr-defined]
            sys.modules[PLUGIN_PACKAGE_ROOT] = root

        name = f"{PLUGIN_PACKAGE_ROOT}.{directory.name}"
        package = ModuleType(name)
        package.__path__ = [str(directory)]  # type: ignore[attr-defined]
        sys.modules[name] = package
        setattr(sys.modules[PLUGIN_PACKAGE_ROOT], directory.name, package)
        return name

    @staticmethod
    def _find_plugin_class(
        module: ModuleType, directory: Path
    ) -> tuple[type[Plugin] | None, list[Message]]:
        """Find the one plugin class the module defines.

        Only classes **defined in this module** count, so the base a plugin
        imports at the top of its ``module.py`` is not mistaken for the plugin
        itself.

        Args:
            module: the imported ``module.py``.
            directory: the plugin's folder, named in any error message.

        Returns:
            A ``(class, messages)`` pair; the class is ``None`` when the
            module holds no plugin class or more than one.
        """
        found = [
            obj
            for obj in vars(module).values()
            if isinstance(obj, type)
            and issubclass(obj, Plugin)
            and obj.__module__ == module.__name__
            and not inspect.isabstract(obj)
        ]
        if not found:
            return None, [
                ErrorMessage(
                    content=(
                        f"plugins/{directory.name}/{PLUGIN_MODULE_FILENAME}: "
                        "defines no plugin class; plugin ignored."
                    )
                )
            ]
        if len(found) > 1:
            names = ", ".join(sorted(obj.__name__ for obj in found))
            return None, [
                ErrorMessage(
                    content=(
                        f"plugins/{directory.name}/{PLUGIN_MODULE_FILENAME}: defines "
                        f"more than one plugin class ({names}); plugin ignored."
                    )
                )
            ]
        return found[0], []
