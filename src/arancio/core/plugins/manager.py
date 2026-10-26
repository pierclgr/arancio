"""Import plugin declarations and attach their hooks and tools at startup."""

import importlib.util
import sys
from collections import Counter
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from arancio.core.constants.path import PLUGIN_MODULE_FILENAME, PLUGINS_PATH
from arancio.core.constants.plugins import PLUGIN_PACKAGE_ROOT
from arancio.core.hooks.base import Hook
from arancio.core.hooks.manager import HookManager
from arancio.core.messages import ErrorMessage, Message
from arancio.core.permissions.types import PermissionCategory
from arancio.core.plugins.base import Plugin
from arancio.core.plugins.registry import PLUGIN_REGISTRY, REGISTRATION_MESSAGES
from arancio.core.plugins.tool import CUSTOM


class PluginManager:
    """Load the shared registry and attach enabled plugins to the hook manager.

    Decorators own registration, including classes imported from helpers. Each
    enabled, non-conflicting plugin is constructed once. Disabled plugins and
    constructor failures remain in the registry with no instance. A failing hook
    method is disabled across its events without stopping sibling methods.

    Tests must repoint this module's PLUGINS_PATH to avoid importing real plugins.
    The root is never created by the manager.

    Attributes:
        registry: the shared list filled by plugin decorators.
        _root: the directory containing plugin entry folders.
        _hook_manager: the shared hook manager plugins are attached to.
        _disabled: instance/function pairs disabled after a hook failure.
    """

    def __init__(self, hook_manager: HookManager) -> None:
        """Keep the shared registry and the hook manager to attach it to.

        Args:
            hook_manager: the hook manager receiving bound plugin methods.
        """
        self.registry: list[Plugin] = PLUGIN_REGISTRY
        self._root: Path = PLUGINS_PATH
        self._hook_manager: HookManager = hook_manager
        self._disabled: set[tuple[int, Callable]] = set()

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the manager.

        Returns:
            The class name followed by the plugins root.
        """
        return f"{type(self).__name__}({self._root})"

    def load(self) -> list[Message]:
        """Import entry files, construct plugins and attach their declarations.

        Metadata errors skip only their class. Import failures discard that
        import's declarations and cached plugin modules. Previously imported
        entries and constructed instances are reused on subsequent calls.

        Returns:
            Startup errors and warnings, including decorator diagnostics.
        """
        messages: list[Message] = []
        if self._root.is_dir():
            for directory in sorted(self._root.iterdir(), key=lambda path: path.name):
                if (
                    not directory.is_dir()
                    or directory.name.startswith((".", "_"))
                    or not (directory / PLUGIN_MODULE_FILENAME).is_file()
                ):
                    continue
                messages.extend(self._import_module(directory))
        messages.extend(REGISTRATION_MESSAGES)
        REGISTRATION_MESSAGES.clear()

        name_counts = Counter(
            plugin.name.upper() for plugin in self.registry if plugin.enabled
        )
        for plugin in self.registry:
            if not plugin.enabled or plugin.instance is not None:
                continue
            if name_counts[plugin.name.upper()] > 1:
                messages.append(
                    ErrorMessage(
                        content=(
                            f"{plugin.location}: duplicate plugin name "
                            f"{plugin.name!r}; "
                            "all plugins with this name are ignored."
                        )
                    )
                )
                continue
            try:
                plugin.instance = plugin.plugin_cls()
            except Exception as exc:
                messages.append(
                    ErrorMessage(
                        content=f"{plugin.location}: cannot be constructed: {exc}"
                    )
                )
                continue
            plugin.hooks = [
                (method.__get__(plugin.instance, plugin.plugin_cls), hooks)
                for method, hooks in plugin.hooks
            ]
            for tool_class in plugin.tools:
                tool_class._method = tool_class._method.__get__(
                    plugin.instance, plugin.plugin_cls
                )
            messages.extend(self._register(plugin))
        return messages

    def _import_module(self, directory: Path) -> list[Message]:
        """Import an entry within its own package, rolling back failed imports.

        Args:
            directory: the folder holding plugin.py and its relative helpers.

        Returns:
            An import error, or an empty list for a successful or cached import.
        """
        package_name = f"{PLUGIN_PACKAGE_ROOT}.{directory.name}"
        module_name = f"{package_name}.{Path(PLUGIN_MODULE_FILENAME).stem}"
        if module_name in sys.modules:
            return []
        registry_start = len(self.registry)
        messages_start = len(REGISTRATION_MESSAGES)
        previous_modules = set(sys.modules)
        try:
            if PLUGIN_PACKAGE_ROOT not in sys.modules:
                root = ModuleType(PLUGIN_PACKAGE_ROOT)
                root.__path__ = []
                sys.modules[PLUGIN_PACKAGE_ROOT] = root
            if package_name not in sys.modules:
                package = ModuleType(package_name)
                package.__path__ = [str(directory)]
                sys.modules[package_name] = package
                setattr(sys.modules[PLUGIN_PACKAGE_ROOT], directory.name, package)
            spec = importlib.util.spec_from_file_location(
                module_name, directory / PLUGIN_MODULE_FILENAME
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            del self.registry[registry_start:]
            del REGISTRATION_MESSAGES[messages_start:]
            # remove helpers too, so retrying imports runs their decorators again
            added_modules = set(sys.modules) - previous_modules
            for name in sorted(added_modules, key=len, reverse=True):
                if name != PLUGIN_PACKAGE_ROOT and not name.startswith(
                    f"{PLUGIN_PACKAGE_ROOT}."
                ):
                    continue
                removed = sys.modules.pop(name, None)
                parent_name, _, child_name = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None and getattr(parent, child_name, None) is removed:
                    delattr(parent, child_name)
            return [
                ErrorMessage(
                    content=(
                        f"plugins/{directory.name}/{PLUGIN_MODULE_FILENAME}: "
                        f"failed to import: {exc}"
                    )
                )
            ]
        return []

    def _register(self, plugin: Plugin) -> list[Message]:
        """Attach the bound hooks and tools of a constructed plugin.

        Args:
            plugin: the record whose bound methods share one instance.

        Returns:
            Errors for tools naming unknown permission categories.
        """
        messages: list[Message] = []
        for method, hooks in plugin.hooks:
            for hook in hooks:
                self._hook_manager.register(
                    hook, partial(self._run_hook, plugin, method, hook)
                )
        for tool_class in plugin.tools:
            if tool_class.category == CUSTOM:
                category = PermissionCategory.get_or_create(f"PLUGIN:{plugin.name}")
            else:
                category = PermissionCategory.get(tool_class.category)
                if category is None:
                    messages.append(
                        ErrorMessage(
                            content=(
                                f"{plugin.location}: tool {tool_class.tool_name!r} "
                                "names unknown permission category "
                                f"{tool_class.category!r}; tool ignored."
                            )
                        )
                    )
                    continue
            category.add_tool(tool_class)
        return messages

    def _run_hook(
        self, plugin: Plugin, method: Callable, hook: Hook, context: dict[str, Any]
    ) -> Message | None:
        """Run a bound method, reporting and disabling it after its first failure.

        Args:
            plugin: the record used to identify the plugin in errors.
            method: the method already bound to the plugin instance.
            hook: the event used to identify a failing dispatch.
            context: the hook manager's shared context dictionary.

        Returns:
            The method's result, or one error when it first fails.
        """
        key = (id(plugin.instance), method.__func__)
        if key in self._disabled:
            return
        try:
            return method(context)
        except Exception as exc:
            self._disabled.add(key)
            return ErrorMessage(
                content=(
                    f"Plugin {plugin.name!r} failed on {hook.value} in "
                    f"{method.__name__}; method disabled: {exc}"
                )
            )
