"""Discovery, loading and attachment of the plugins found on disk."""

from functools import partial
from pathlib import Path
from typing import Any

from arancio.core.constants.path import PLUGIN_MANIFEST_FILENAME, PLUGINS_PATH
from arancio.core.hooks.manager import HookManager
from arancio.core.hooks.types import Hook
from arancio.core.messages import ErrorMessage, Message
from arancio.core.permissions.types import PermissionCategory
from arancio.core.plugins.base import Plugin
from arancio.core.plugins.loader import PluginLoader
from arancio.core.plugins.tool import CUSTOM, build_plugin_tool_class, find_tool_specs


class PluginManager:
    """Find the plugins on disk, load them and attach them to the hook manager.

    A plugin that raises is disabled across all its hooks for the rest of the
    process. Its wrapper returns one ErrorMessage through the hook caller's
    normal message stream, without interrupting other plugins or retrying the
    agent. Ordinary hook handler exceptions still propagate.

    Like the tool harness, the plugins directory is read straight from the
    module constant :data:`~arancio.core.constants.path.PLUGINS_PATH` rather
    than through a :class:`~arancio.storage.manager.StorageManager`, and the
    user places their folders there by hand — the directory is never created.
    A test must therefore repoint ``manager.PLUGINS_PATH`` on **this** module,
    exactly as it repoints ``tools_base.TOOLS_HARNESS_PATH``, or it scans the
    user's real plugins.

    Attributes:
        _root: the directory the plugin folders are discovered in.
        _hook_manager: the shared hook manager plugins are registered on.
        _plugins: the plugins that loaded and are enabled.
        _disabled: the plugins switched off after raising at runtime.
    """

    def __init__(self, hook_manager: HookManager) -> None:
        """Initialize the manager with what to attach the plugins to.

        Args:
            hook_manager: the shared hook manager
                :class:`~arancio.core.plugins.base.Plugin` instances are
                registered on.
        """
        self._root: Path = PLUGINS_PATH
        self._hook_manager = hook_manager
        self._plugins: list[Plugin] = []
        self._disabled: set[Plugin] = set()

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the manager.

        Returns:
            The class name followed by the plugins root.
        """
        return f"{type(self).__name__}({self._root})"

    @property
    def plugins(self) -> list[Plugin]:
        """Return the plugins that loaded and are enabled.

        Returns:
            The loaded plugins, in discovery order.
        """
        return list(self._plugins)

    def load(self) -> list[Message]:
        """Discover, load and register every plugin under the plugins root.

        Mirrors :meth:`~arancio.settings.manager.SettingsManager.load`: it
        never raises, and returns the messages the UI shows at startup. A
        broken plugin folder costs one message and is skipped; the rest still
        load.

        Returns:
            The messages describing every folder that could not be loaded or
            whose manifest needed a fallback, in discovery order.
        """
        messages: list[Message] = []
        for directory in self._discover():
            plugin, plugin_messages = PluginLoader.load(directory)
            messages.extend(plugin_messages)
            if plugin is None:
                continue
            self._plugins.append(plugin)
            messages.extend(self._register(plugin))
        return messages

    def _discover(self) -> list[Path]:
        """List the plugin folders under the plugins root.

        Only immediate subdirectories count, and only those holding a manifest
        — the manifest is what makes a folder a plugin, so a folder without one
        is not a broken plugin, it is not a plugin. Names starting with ``.``
        or ``_`` are skipped outright, which keeps ``__pycache__`` and editor
        droppings out.

        Returns:
            The plugin folders, sorted by name. Empty when the plugins
            directory does not exist — nothing creates it, so a user who has
            never added a plugin simply has none.
        """
        if not self._root.is_dir():
            return []
        return sorted(
            (
                directory
                for directory in self._root.iterdir()
                if directory.is_dir()
                and not directory.name.startswith((".", "_"))
                and (directory / PLUGIN_MANIFEST_FILENAME).is_file()
            ),
            key=lambda directory: directory.name,
        )

    def _register(self, plugin: Plugin) -> list[Message]:
        """Register a loaded plugin's hooks and tools.

        Args:
            plugin: the plugin to attach.

        Returns:
            One error per tool spec naming an unknown category, in
            definition order.
        """
        messages: list[Message] = []
        for hook in plugin.hooks:
            self._hook_manager.register(hook, partial(self._run_plugin, plugin, hook))

        for func, spec in find_tool_specs(type(plugin)):
            if spec.category == CUSTOM:
                category = PermissionCategory.get_or_create(
                    f"PLUGIN:{type(plugin).__name__}"
                )
            else:
                category = PermissionCategory.get(spec.category)
                if category is None:
                    messages.append(
                        ErrorMessage(
                            content=(
                                f"plugins/{plugin.directory.name}: tool {spec.name!r} "
                                f"names unknown permission category "
                                f"{spec.category!r}; tool ignored."
                            )
                        )
                    )
                    continue
            category.add_tool(build_plugin_tool_class(plugin, func, spec))
        return messages

    def _run_plugin(
        self, plugin: Plugin, hook: Hook, **kwargs: Any
    ) -> ErrorMessage | None:
        """Run one plugin for one dispatch, disabling it if it raises.

        ``plugin`` and ``hook`` are bound through
        :func:`functools.partial`, so the plugin learns which hook fired:
        :meth:`~arancio.core.hooks.manager.HookManager.run` forwards only the
        hook's own keyword arguments and never names the hook itself. A plugin
        already disabled is skipped, so it never runs again.

        Args:
            plugin: the plugin to run.
            hook: the hook this dispatch is for.
            **kwargs: the hook's own keyword arguments, unchanged.

        Returns:
            One error when the plugin fails, otherwise None.
        """
        if plugin in self._disabled:
            return
        try:
            plugin.execute(hook=hook, **kwargs)
        except Exception as exc:
            self._disabled.add(plugin)
            return ErrorMessage(
                content=(
                    f"Plugin {plugin.name!r} failed on {hook.value} and was disabled: "
                    f"{exc}"
                )
            )
