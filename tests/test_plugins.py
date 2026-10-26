"""Tests for plugin registration, loading, hook attachment and failure isolation."""

import asyncio
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from fakes import RecordingApp, ScriptedClient, ScriptedController
from textual.widgets import Input, Markdown

from arancio.core.agents import Agent
from arancio.core.builders.payload.litellm import LiteLLMPayloadBuilder
from arancio.core.hooks.base import Hook
from arancio.core.hooks.manager import HookManager
from arancio.core.messages import (
    AssistantMessage,
    ErrorMessage,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
    WarningMessage,
)
from arancio.core.permissions.manager import PermissionManager
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.core.plugins.base import Plugin
from arancio.core.plugins.decorators import plugin
from arancio.core.plugins.manager import PluginManager
from arancio.core.plugins.registry import PLUGIN_REGISTRY
from arancio.core.plugins.tool import PluginTool
from arancio.core.tools.manager import ToolManager
from arancio.prompt.actions.executor import ActionExecutor
from arancio.prompt.actions.types import PromptAction, ShellCommandAction
from arancio.sessions.manager import SessionManager
from arancio.settings.manager import SettingsManager
from arancio.settings.validator import SettingsValidator
from arancio.storage.manager import StorageManager
from arancio.ui.app import App

_LOGGER_MODULE = """
from arancio.core.hooks.base import Hook
from arancio.core.plugins.decorators import hook, plugin

calls = []


@plugin()
class Logger:
    @hook(Hook.AGENT_START)
    def on_agent_start(self, context) -> None:
        calls.append(context["message"])
"""

_SECOND_CLASS = """

@plugin()
class Second:
    @hook(Hook.AGENT_END)
    def on_agent_end(self, context) -> None:
        pass
"""


def _write_plugin(
    root: Path,
    name: str,
    manifest: str | None = None,
    module: str | None = _LOGGER_MODULE,
    extra_files: dict[str, str] | None = None,
) -> Path:
    """Create one plugin folder on disk.

    Args:
        root: the plugins root the folder is created under.
        name: the plugin's folder name.
        manifest: an obsolete manifest body, or ``None`` to leave it out.
        module: the ``plugin.py`` body, or ``None`` to leave it out.
        extra_files: further files to drop in the folder, keyed by filename.

    Returns:
        The created plugin folder.
    """
    directory = root / name
    directory.mkdir(parents=True)
    if manifest is not None:
        (directory / "manifest.yml").write_text(manifest)
    if module is not None:
        (directory / "plugin.py").write_text(textwrap.dedent(module))
    for filename, content in (extra_files or {}).items():
        (directory / filename).write_text(textwrap.dedent(content))
    return directory


def _loaded_module(name: str) -> Any:
    """Return a loaded plugin's module, to read what its plugin recorded.

    Args:
        name: the plugin's folder name.

    Returns:
        The module object the manager put in ``sys.modules``.
    """
    return sys.modules[f"arancio_plugins.{name}.plugin"]


@pytest.fixture
def plugins_root(tmp_path: Path) -> Path:
    """Return the plugins directory the autouse fixture repointed at ``tmp_path``.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The created, empty plugins directory.
    """
    root = tmp_path / "plugins"
    root.mkdir()
    return root


@pytest.fixture
def plugin_manager(hook_manager: HookManager) -> PluginManager:
    """Return a plugin manager reading from the temporary plugins root.

    Args:
        hook_manager: the hook manager plugins are registered on.

    Returns:
        A plugin manager that has not loaded anything yet.
    """
    return PluginManager(hook_manager=hook_manager)


def test_decorators_register_defaults_without_constructing_instances() -> None:
    """Registration keeps declarations lazy and shares the manager's plain list."""

    @plugin()
    class Plain:
        """A plugin whose constructor must wait until startup."""

        def __init__(self) -> None:
            """Reject construction during decoration.

            Raises:
                RuntimeError: if decoration constructs the plugin.
            """
            raise RuntimeError("constructed too early")

    manager = PluginManager(hook_manager=HookManager())
    assert isinstance(PLUGIN_REGISTRY, list)
    assert manager.registry is PLUGIN_REGISTRY
    (record,) = manager.registry
    assert record.plugin_cls is Plain
    assert record.name == "Plain"
    assert record.version == "0.0.0"
    assert record.description == ""
    assert record.author == ""
    assert record.enabled is True
    assert record.instance is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", 123),
        ("name", None),
        ("name", ""),
        ("version", 1),
        ("description", None),
        ("author", False),
        ("enabled", "False"),
        ("enabled", 1),
        ("bogus", "unknown"),
        ("instance", "injected"),
        ("hooks", []),
    ],
)
def test_bad_metadata_skips_only_its_class(
    plugins_root: Path, plugin_manager: PluginManager, field: str, value: Any
) -> None:
    """Wrong types and unknown fields report errors while sibling classes load."""
    module = (
        _LOGGER_MODULE
        + f"""
@plugin(**{{{field!r}: {value!r}}})
class Invalid:
    def __init__(self):
        raise RuntimeError("invalid plugin was constructed")
"""
        + _SECOND_CLASS
    )
    _write_plugin(plugins_root, "bad_metadata", module=module)

    messages = plugin_manager.load()

    assert len(messages) == 1
    assert isinstance(messages[0], ErrorMessage)
    assert "plugin.py:Invalid" in messages[0].content
    assert field in messages[0].content
    assert repr(value) in messages[0].content
    assert [item.name for item in plugin_manager.registry] == ["Logger", "Second"]
    assert all(item.instance is not None for item in plugin_manager.registry)
    assert plugin_manager.load() == []


def test_plugin_records_reject_wrong_metadata_types(tmp_path: Path) -> None:
    """Record construction enforces metadata types without a separate validator."""
    with pytest.raises(TypeError, match="enabled.*'False'"):
        Plugin(
            plugin_cls=object,
            source_path=tmp_path / "plugin.py",
            name="Invalid",
            enabled="False",
        )


def test_plugin_metadata_overrides_are_kept(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """Explicit metadata stays on the record without changing the class."""
    module = _LOGGER_MODULE.replace(
        "@plugin()",
        "@plugin(name='Named', version='1.2.3', "
        "description='Example', author='Author')",
    )
    _write_plugin(plugins_root, "metadata_overrides", module=module)

    assert plugin_manager.load() == []
    (record,) = plugin_manager.registry
    assert (record.name, record.version, record.description, record.author) == (
        "Named",
        "1.2.3",
        "Example",
        "Author",
    )
    assert record.plugin_cls.__name__ == "Logger"


@pytest.mark.parametrize("enabled", [True, False])
def test_empty_plugins_are_registered_and_warn(
    plugins_root: Path, plugin_manager: PluginManager, enabled: bool
) -> None:
    """Empty declarations warn once, and only enabled ones are constructed."""
    _write_plugin(
        plugins_root,
        "empty_declaration",
        module=f"""
from arancio.core.plugins.decorators import plugin
constructed = []

@plugin(enabled={enabled!r})
class Empty:
    def __init__(self):
        constructed.append(self)
""",
    )

    messages = plugin_manager.load()

    assert len(messages) == 1
    assert isinstance(messages[0], WarningMessage)
    assert "plugin.py:Empty" in messages[0].content
    assert "no hooks and no tools" in messages[0].content
    (record,) = plugin_manager.registry
    assert record.enabled is enabled
    assert (record.instance is not None) is enabled
    assert _loaded_module("empty_declaration").constructed == (
        [record.instance] if enabled else []
    )
    assert plugin_manager.load() == []


def test_an_entry_without_declarations_is_ignored(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """Importing an entry with no decorated classes produces no registry entries."""
    _write_plugin(plugins_root, "no_declarations", module="class Helper: pass")

    assert plugin_manager.load() == []
    assert plugin_manager.registry == []


def test_failed_import_discards_partial_registrations_and_can_be_retried(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """A failed import rolls back declarations, diagnostics and cached helpers."""
    directory = _write_plugin(
        plugins_root,
        "aaa_partial_import",
        module="""
from .helpers import Helper
from arancio.core.plugins.decorators import plugin

@plugin()
class Empty: pass

@plugin(enabled='wrong')
class Invalid: pass

raise RuntimeError('import stopped')
""",
        extra_files={
            "helpers.py": _LOGGER_MODULE.replace("Logger", "Helper"),
        },
    )
    _write_plugin(plugins_root, "zzz_after_failure")

    messages = plugin_manager.load()

    assert len(messages) == 1
    assert "import stopped" in messages[0].content
    assert [item.name for item in plugin_manager.registry] == ["Logger"]
    assert not any(
        name.startswith("arancio_plugins.aaa_partial_import") for name in sys.modules
    )
    (directory / "plugin.py").write_text("from .helpers import Helper\n")

    assert plugin_manager.load() == []
    assert [item.name for item in plugin_manager.registry] == ["Logger", "Helper"]
    message = UserMessage(content="retry")
    hook_manager.run(Hook.AGENT_START, message=message)
    assert _loaded_module("zzz_after_failure").calls == [message]
    helper_module = sys.modules["arancio_plugins.aaa_partial_import.helpers"]
    assert helper_module.calls == [message]


def test_declaration_order_and_bound_methods_are_preserved(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """Loading binds methods once, preserving folder, class and method order."""
    module = """
from arancio.core.plugins.decorators import plugin, tool

@plugin(name={name!r})
class Ordered:
    @tool(description='First', input_schema={{}})
    def z_first(self): return 'first'

    @tool(description='Second', input_schema={{}})
    def a_second(self): return 'second'
"""
    _write_plugin(plugins_root, "zzz_order", module=module.format(name="Last"))
    _write_plugin(plugins_root, "aaa_order", module=module.format(name="First"))

    assert plugin_manager.load() == []
    assert [item.name for item in plugin_manager.registry] == ["First", "Last"]
    for record in plugin_manager.registry:
        assert [tool.tool_name for tool in record.tools] == ["z_first", "a_second"]
        assert [tool.__name__ for tool in record.tools] == [
            f"{record.name}-z_first",
            f"{record.name}-a_second",
        ]
        assert all(tool._method.__self__ is record.instance for tool in record.tools)
    assert plugin_manager.load() == []
    assert len(PermissionCategory.PLUGIN.tools) == 4
    assert hook_manager.run(Hook.STARTUP) == []


@pytest.mark.parametrize(
    "manifest", ["just a string", "name: [unclosed", "enabled: false"]
)
def test_obsolete_manifests_are_ignored(
    plugins_root: Path, plugin_manager: PluginManager, manifest: str
) -> None:
    """YAML files no longer control discovery, metadata or enabled status."""
    _write_plugin(plugins_root, "obsolete_manifest", manifest=manifest)

    assert plugin_manager.load() == []
    assert [item.name for item in plugin_manager.registry] == ["Logger"]


def test_plain_plugin_instances_receive_no_runtime_attributes(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """Plain instances receive no metadata, manifest, directory or context."""
    _write_plugin(plugins_root, "plain_instance")

    assert plugin_manager.load() == []

    instance = plugin_manager.registry[0].instance
    assert type(instance).__bases__ == (object,)
    assert vars(instance) == {}
    for owner in (type(instance), instance):
        assert not any(
            hasattr(owner, field)
            for field in (
                "directory",
                "metadata",
                "manifest",
                "plugin_context",
                "name",
                "__plugin_metadata",
                "__arancio_plugin__",
            )
        )


def test_a_plugin_file_without_a_manifest_is_loaded(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """Only plugin.py is required to discover the folder."""
    directory = _write_plugin(plugins_root, "no_manifest")

    assert not (directory / "manifest.yml").exists()
    assert plugin_manager.load() == []
    assert [item.name for item in plugin_manager.registry] == ["Logger"]


def test_underscore_and_dot_prefixed_folders_are_skipped(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """``__pycache__`` and hidden folders never count as plugins."""
    _write_plugin(plugins_root, "__pycache__")
    _write_plugin(plugins_root, ".hidden")

    assert plugin_manager.load() == []
    assert plugin_manager.registry == []


def test_a_missing_plugins_root_reports_nothing(
    tmp_path: Path, plugin_manager: PluginManager
) -> None:
    """Nothing creates the directory, so not having one just means no plugins."""
    assert not (tmp_path / "plugins").exists()

    assert plugin_manager.load() == []
    assert plugin_manager.registry == []


@pytest.mark.parametrize(
    "legacy_module", [False, True], ids=["missing", "old-filename"]
)
def test_folders_without_plugin_py_are_skipped(
    plugins_root: Path, plugin_manager: PluginManager, legacy_module: bool
) -> None:
    """A manifest or the old entry filename does not make a folder eligible."""
    directory = _write_plugin(
        plugins_root, "no_plugin_entry", module=None, manifest="enabled: true"
    )
    if legacy_module:
        (directory / "module.py").write_text(_LOGGER_MODULE)

    assert plugin_manager.load() == []
    assert plugin_manager.registry == []


def test_plugin_py_is_loaded_under_its_matching_module_name(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """The required entry file loads as the plugin package's ``plugin`` module."""
    from arancio.core.constants.plugins import PLUGIN_PACKAGE_ROOT

    assert PLUGIN_PACKAGE_ROOT == "arancio_plugins"
    directory = _write_plugin(plugins_root, "named_entry")

    assert plugin_manager.load() == []

    plugin = plugin_manager.registry[0]
    assert type(plugin.instance).__module__ == "arancio_plugins.named_entry.plugin"
    assert Path(_loaded_module("named_entry").__file__) == directory / "plugin.py"


def test_a_module_that_fails_to_import_is_reported_and_others_still_load(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """One broken folder costs one message and never blocks its siblings."""
    _write_plugin(plugins_root, "aaa_broken", module="raise RuntimeError('boom')")
    _write_plugin(plugins_root, "zzz_healthy")

    messages = plugin_manager.load()

    assert len(messages) == 1
    assert isinstance(messages[0], ErrorMessage)
    assert "failed to import" in messages[0].content
    assert [plugin.source_path.parent.name for plugin in plugin_manager.registry] == [
        "zzz_healthy"
    ]


def test_a_file_can_define_multiple_independent_plugins(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """Each decorated class becomes its own plugin in definition order."""
    _write_plugin(plugins_root, "two_classes", module=_LOGGER_MODULE + _SECOND_CLASS)

    assert plugin_manager.load() == []

    loaded = plugin_manager.registry
    assert [item.name for item in loaded] == ["Logger", "Second"]
    assert loaded[0].version == loaded[1].version
    assert loaded[0].instance is not loaded[1].instance
    assert loaded[0].source_path == loaded[1].source_path


def test_a_tools_only_plugin_with_no_hooks_loads_successfully(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """A plugin defining only a @tool method no longer needs any hooks."""
    _write_plugin(
        plugins_root,
        "tools_only",
        module="""
        from arancio.core.plugins.decorators import plugin, tool


        @plugin()
        class Idle:
            @tool(description="Does nothing.", input_schema={"properties": {}})
            def noop(self) -> None:
                pass
        """,
    )

    messages = plugin_manager.load()

    assert messages == []
    assert [type(plugin.instance).__name__ for plugin in plugin_manager.registry] == [
        "Idle"
    ]


def test_a_disabled_plugin_is_imported_but_not_constructed_or_registered(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """Disabled metadata skips creation after the module has been imported."""
    module = _LOGGER_MODULE.replace("@plugin()", "@plugin(enabled=False)")
    module += "\nLogger.__init__ = lambda self: calls.append('constructed')\n"
    _write_plugin(plugins_root, "switched_off", module=module)

    assert plugin_manager.load() == []
    (record,) = plugin_manager.registry
    assert record.enabled is False
    assert record.instance is None
    assert _loaded_module("switched_off").calls == []
    hook_manager.run(Hook.AGENT_START, message=UserMessage(content="hi"))
    assert _loaded_module("switched_off").calls == []


def test_a_plugin_can_import_a_sibling_module_from_its_own_folder(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """The synthetic package makes a plugin's own modules reachable."""
    _write_plugin(
        plugins_root,
        "with_helpers",
        module="""
        from arancio.core.hooks.base import Hook
        from arancio.core.plugins.decorators import hook, plugin

        from pathlib import Path

        from .helpers import GREETING


        @plugin()
        class Greeter:
            @hook(Hook.AGENT_START)
            def greet(self, context) -> None:
                (Path(__file__).resolve().parent / "out.txt").write_text(GREETING)
        """,
        extra_files={"helpers.py": "GREETING = 'hello'\n"},
    )

    assert plugin_manager.load() == []

    hook_manager.run(Hook.AGENT_START, message=UserMessage(content="hi"))

    assert (plugins_root / "with_helpers" / "out.txt").read_text() == "hello"


def test_a_registered_plugin_runs_on_dispatch_with_the_hook_context(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """The hook method receives the shared context holding the hook's data."""
    directory = _write_plugin(plugins_root, "records_calls")

    assert plugin_manager.load() == []
    assert [plugin.name for plugin in plugin_manager.registry] == ["Logger"]
    assert plugin_manager.registry[0].source_path == directory / "plugin.py"

    message = UserMessage(content="hi")
    hook_manager.run(Hook.AGENT_START, message=message)

    assert _loaded_module("records_calls").calls == [message]


def test_two_hook_methods_each_run_only_on_their_own_hook(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """One plugin can bind a different method to each hook."""
    _write_plugin(
        plugins_root,
        "two_methods",
        module="""
        from arancio.core.hooks.base import Hook
        from arancio.core.plugins.decorators import hook, plugin

        seen = []


        @plugin()
        class Split:
            @hook(Hook.TURN_START)
            def on_start(self, context) -> None:
                seen.append(("start", context["turn"]))

            @hook(Hook.TURN_END)
            def on_end(self, context) -> None:
                seen.append(("end", context["turn"]))
        """,
    )

    assert plugin_manager.load() == []

    hook_manager.run(Hook.TURN_START, turn=0)
    hook_manager.run(Hook.TURN_END, turn=1)

    assert _loaded_module("two_methods").seen == [("start", 0), ("end", 1)]


def test_a_method_with_stacked_hooks_runs_on_each_of_them(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """Stacking ``@hook`` binds one method to several hooks."""
    _write_plugin(
        plugins_root,
        "stacked_hooks",
        module="""
        from arancio.core.hooks.base import Hook
        from arancio.core.plugins.decorators import hook, plugin

        seen = []


        @plugin()
        class Both:
            @hook(Hook.TURN_START)
            @hook(Hook.TURN_END)
            def on_turn(self, context) -> None:
                seen.append(context["turn"])
        """,
    )

    assert plugin_manager.load() == []

    hook_manager.run(Hook.TURN_START, turn=0)
    hook_manager.run(Hook.TURN_END, turn=1)

    assert _loaded_module("stacked_hooks").seen == [0, 1]


def test_a_failing_plugin_does_not_stop_the_plugins_behind_it(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """The wrapper catches, so ``HookManager``'s stop-on-raise never applies.

    Plugins are registered in folder order, so ``aaa_raises`` runs first.
    """
    _write_plugin(
        plugins_root,
        "aaa_raises",
        module="""
        from arancio.core.hooks.base import Hook
        from arancio.core.plugins.decorators import hook, plugin


        @plugin()
        class Broken:
            @hook(Hook.AGENT_START)
            def on_agent_start(self, context) -> None:
                raise RuntimeError("boom")
        """,
    )
    _write_plugin(plugins_root, "zzz_records")

    assert plugin_manager.load() == []

    hook_manager.run(Hook.AGENT_START, message=UserMessage(content="hi"))

    assert len(_loaded_module("zzz_records").calls) == 1


def test_a_failing_plugin_is_disabled_and_reported_once(
    plugins_root: Path,
    plugin_manager: PluginManager,
    hook_manager: HookManager,
) -> None:
    """A raising hook method never runs again, and the user is told exactly once."""
    _write_plugin(
        plugins_root,
        "always_raises",
        module="""
        from arancio.core.hooks.base import Hook
        from arancio.core.plugins.decorators import hook, plugin

        runs = []


        @plugin()
        class Broken:
            @hook(Hook.AGENT_START)
            def explode(self, context) -> None:
                runs.append(1)
                raise RuntimeError("boom")
        """,
    )

    assert plugin_manager.load() == []

    notices = hook_manager.run(Hook.AGENT_START, message=UserMessage(content="one"))
    assert hook_manager.run(Hook.AGENT_START, message=UserMessage(content="two")) == []

    assert _loaded_module("always_raises").runs == [1]

    assert len(notices) == 1
    assert isinstance(notices[0], ErrorMessage)
    assert not notices[0].in_history
    assert "Broken" in notices[0].content
    assert "explode" in notices[0].content
    assert "boom" in notices[0].content


def test_a_failing_method_stops_on_all_its_hooks_but_siblings_keep_running(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """Disabling is per method: its stacked hooks stop, other methods do not."""
    _write_plugin(
        plugins_root,
        "fails_then_quiet",
        module="""
        from arancio.core.hooks.base import Hook
        from arancio.core.plugins.decorators import hook, plugin

        broken_runs = []
        healthy_runs = []


        @plugin()
        class Mixed:
            @hook(Hook.TURN_START)
            @hook(Hook.TURN_END)
            def broken(self, context) -> None:
                broken_runs.append(context["turn"])
                raise RuntimeError("boom")

            @hook(Hook.TURN_END)
            def healthy(self, context) -> None:
                healthy_runs.append(context["turn"])
        """,
    )

    assert plugin_manager.load() == []

    hook_manager.run(Hook.TURN_START, turn=0)
    hook_manager.run(Hook.TURN_END, turn=0)
    hook_manager.run(Hook.TURN_END, turn=1)

    module = _loaded_module("fails_then_quiet")
    assert module.broken_runs == [0]
    assert module.healthy_runs == [0, 1]


def test_a_failing_plugin_does_not_break_an_agent_run(
    plugins_root: Path, plugin_manager: PluginManager, agent: Agent
) -> None:
    """Isolation is what keeps a broken plugin from ending the whole run.

    Without it the exception would escape ``BaseTool.call`` into the agent's single
    ``except``, be labelled ``source="model"`` and retry the turn.
    """
    _write_plugin(
        plugins_root,
        "breaks_turns",
        module="""
        from arancio.core.hooks.base import Hook
        from arancio.core.plugins.decorators import hook, plugin


        @plugin()
        class Broken:
            @hook(Hook.TURN_START)
            def on_turn_start(self, context) -> None:
                raise RuntimeError("boom")
        """,
    )
    assert plugin_manager.load() == []

    produced = list(agent(message=UserMessage(content="hi")))

    assert [type(message) for message in produced] == [ErrorMessage, AssistantMessage]


def test_a_plugin_hook_message_is_displayed_and_restored_outside_model_history(
    plugins_root: Path,
    plugin_manager: PluginManager,
    agent: Agent,
    client: ScriptedClient,
    hook_manager: HookManager,
    settings_manager: SettingsManager,
    session_manager: SessionManager,
    storage_manager: StorageManager,
) -> None:
    """A hook greeting reaches the chat log and session without reaching the model."""
    _write_plugin(
        plugins_root,
        "ui_greeter",
        module="""
        from arancio.core.hooks.base import Hook
        from arancio.core.messages import AssistantMessage
        from arancio.core.plugins.decorators import hook, plugin


        @plugin()
        class Greeter:
            @hook(Hook.AGENT_START)
            def greet(self, context):
                return AssistantMessage(content="Hello World", in_history=False)
        """,
    )
    assert plugin_manager.load() == []
    settings_manager.settings.provider = "openai"
    settings_manager.settings.model_name = "gpt-4o"
    app = App(
        agent=agent,
        model_id=client.model_id,
        settings_manager=settings_manager,
        session_manager=session_manager,
        hook_manager=hook_manager,
    )

    async def submit_prompt() -> None:
        """Submit a prompt and check the greeting in the real Textual chat log."""
        test_context = app.run_test()
        async with test_context as pilot:
            app.query_one("#prompt", Input).value = "hi"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            rendered = [
                widget.source for widget in app.query_one("#log").query(Markdown)
            ]
            assert rendered == ["Hello World", "done"]

    asyncio.run(submit_prompt())

    assert len(client.histories) == 1
    assert [message.content for message in client.histories[0]] == ["hi"]
    assert [message.content for message in agent._message_history] == ["hi", "done"]
    reopened = SessionManager(
        storage_manager,
        session_manager.current.configuration,
        root=storage_manager.root,
    )
    reopened.load(session_manager.current.id)
    assert [message.content for message in reopened.visible_messages()] == [
        "hi",
        "Hello World",
        "done",
    ]
    assert [message.content for message in reopened.model_history()] == ["hi", "done"]


@pytest.mark.parametrize("exit_path", ["exit", "quit", "ctrl_c", "direct"])
def test_application_hooks_run_once_without_a_user_prompt(
    plugins_root: Path,
    plugin_manager: PluginManager,
    agent: Agent,
    client: ScriptedClient,
    hook_manager: HookManager,
    settings_manager: SettingsManager,
    session_manager: SessionManager,
    exit_path: str,
) -> None:
    """Application hooks share context, run once and leave an empty chat unsaved."""
    name = f"lifecycle_{exit_path}"
    _write_plugin(
        plugins_root,
        name,
        module="""
        from arancio.core.hooks.base import Hook
        from arancio.core.messages import AssistantMessage
        from arancio.core.plugins.decorators import hook, plugin

        events = []


        @plugin()
        class Lifecycle:
            @hook(Hook.STARTUP)
            def start(self, context):
                events.append(("startup", context))
                return AssistantMessage(content="Hello World", in_history=False)

            @hook(Hook.SHUTDOWN)
            def stop(self, context):
                events.append(("shutdown", context))
        """,
    )
    assert plugin_manager.load() == []
    module = _loaded_module(name)
    app = App(
        agent=agent,
        model_id=client.model_id,
        settings_manager=settings_manager,
        session_manager=session_manager,
        hook_manager=hook_manager,
    )

    async def open_and_exit() -> None:
        """Check the launch greeting before closing through the selected route."""
        test_context = app.run_test()
        async with test_context as pilot:
            assert module.events == [("startup", hook_manager.context)]
            assert module.events[0][1] is hook_manager.context
            assert hook_manager.context["agent"] is agent
            assert hook_manager.context["client"] is client
            assert (
                hook_manager.context["permission_manager"] is agent._permission_manager
            )
            assert hook_manager.context["summary_client"] is client
            rendered = [
                widget.source for widget in app.query_one("#log").query(Markdown)
            ]
            assert rendered == ["Hello World"]
            assert not session_manager.current.path.exists()
            if exit_path in {"exit", "quit"}:
                app.query_one("#prompt", Input).value = f"/{exit_path}"
                await pilot.press("enter")
            elif exit_path == "ctrl_c":
                await pilot.press("ctrl+c")
            else:
                app.exit()

    asyncio.run(open_and_exit())

    assert module.events == [
        ("startup", hook_manager.context),
        ("shutdown", hook_manager.context),
    ]
    assert module.events[1][1] is hook_manager.context
    assert client.requests == []
    assert agent._message_history == []
    assert not session_manager.current.path.exists()


@pytest.mark.parametrize("failing_hook", ["STARTUP", "SHUTDOWN"])
def test_application_hook_failures_do_not_block_start_or_exit(
    plugins_root: Path,
    plugin_manager: PluginManager,
    agent: Agent,
    client: ScriptedClient,
    hook_manager: HookManager,
    settings_manager: SettingsManager,
    session_manager: SessionManager,
    failing_hook: str,
) -> None:
    """A failing lifecycle method does not stop the application from closing."""
    name = f"broken_{failing_hook.lower()}"
    _write_plugin(
        plugins_root,
        name,
        module=f"""
        from arancio.core.hooks.base import Hook
        from arancio.core.plugins.decorators import hook, plugin

        events = []


        @plugin()
        class BrokenLifecycle:
            @hook(Hook.STARTUP)
            def start(self, context):
                events.append("startup")
                if "{failing_hook}" == "STARTUP":
                    raise RuntimeError("startup boom")

            @hook(Hook.SHUTDOWN)
            def stop(self, context):
                events.append("shutdown")
                if "{failing_hook}" == "SHUTDOWN":
                    raise RuntimeError("shutdown boom")
        """,
    )
    assert plugin_manager.load() == []
    app = App(
        agent=agent,
        model_id=client.model_id,
        settings_manager=settings_manager,
        session_manager=session_manager,
        hook_manager=hook_manager,
    )

    async def open_and_exit() -> None:
        """A startup error is visible while shutdown errors still permit exit."""
        test_context = app.run_test()
        async with test_context:
            assert len(app.query("#log .error")) == (failing_hook == "STARTUP")
            app.exit()

    asyncio.run(open_and_exit())

    assert _loaded_module(name).events == ["startup", "shutdown"]
    assert not session_manager.current.path.exists()


@pytest.mark.parametrize(
    "hook", [hook for hook in Hook if hook not in {Hook.STARTUP, Hook.SHUTDOWN}]
)
def test_plugin_failures_reach_agent_without_retry(
    plugins_root: Path,
    plugin_manager: PluginManager,
    agent: Agent,
    client: ScriptedClient,
    hook: Hook,
    tmp_path: Path,
) -> None:
    """Each dispatch forwards plugin failures without losing the tool outcome."""
    name = f"fails_{hook.value}"
    _write_plugin(
        plugins_root,
        name,
        module=f"""
from arancio.core.hooks.base import Hook
from arancio.core.plugins.decorators import hook, plugin
@plugin(name={name!r})
class Broken:
    @hook(Hook.{hook.name})
    def explode(self, context):
        raise RuntimeError("plugin boom")
""",
    )
    assert plugin_manager.load() == []
    client.turns = [
        [
            ToolCallMessage(
                content="read",
                id="c1",
                name="ReadFileTool",
                arguments={"file_path": str(tmp_path / "missing")},
            )
        ],
        [AssistantMessage(content="done")],
    ]
    produced = list(agent(UserMessage(content="hi")))
    errors = [m for m in produced if type(m) is ErrorMessage]
    assert len(errors) == 1
    assert name in errors[0].content
    assert errors[0] not in agent._message_history
    assert sum(isinstance(m, ToolResultMessage) for m in produced) == 1
    assert len(client.requests) == 2


def test_closing_agent_runs_final_hook_without_yielding(
    plugins_root: Path,
    plugin_manager: PluginManager,
    agent: Agent,
) -> None:
    """Closing a suspended run still disables a broken final hook safely."""
    _write_plugin(
        plugins_root,
        "close_turn",
        module="""
from arancio.core.hooks.base import Hook
from arancio.core.plugins.decorators import hook, plugin
runs = []
@plugin()
class Broken:
    @hook(Hook.TURN_END)
    def explode(self, context):
        runs.append(1)
        raise RuntimeError("close boom")
""",
    )
    assert plugin_manager.load() == []
    stream = agent(UserMessage(content="hi"))
    next(stream)
    stream.close()
    assert _loaded_module("close_turn").runs == [1]


@pytest.mark.parametrize(
    "action_kind", ["prompt", "shell", "silent_shell", "file", "directory"]
)
def test_plugin_errors_are_saved_once_and_restored_outside_model_history(
    plugins_root: Path,
    plugin_manager: PluginManager,
    agent: Agent,
    hook_manager: HookManager,
    settings_manager: SettingsManager,
    session_manager: SessionManager,
    storage_manager: StorageManager,
    tmp_path: Path,
    action_kind: str,
) -> None:
    """All executor paths save errors without changing tool or model history."""
    hook = "AGENT_END" if action_kind == "prompt" else "BEFORE_TOOL_CALL"
    _write_plugin(
        plugins_root,
        f"persist_{action_kind}",
        module=f"""
from arancio.core.hooks.base import Hook
from arancio.core.plugins.decorators import hook, plugin
@plugin()
class Broken:
    @hook(Hook.{hook})
    def explode(self, context):
        raise RuntimeError("persist boom")
""",
    )
    assert plugin_manager.load() == []
    settings_manager.settings.provider = "openai"
    settings_manager.settings.model_name = "gpt-4o"
    executor = ActionExecutor(
        agent, RecordingApp(), settings_manager, session_manager, hook_manager
    )
    if action_kind in {"shell", "silent_shell"}:
        action = ShellCommandAction(
            command="echo hello",
            raw_input="!echo hello",
            add_to_history=action_kind == "shell",
        )
    else:
        target = tmp_path / "readme.txt"
        target.write_text("hello")
        mentions = (
            []
            if action_kind == "prompt"
            else [tmp_path if action_kind == "directory" else target]
        )
        action = PromptAction(prompt="hi", raw_input="hi", mentions=mentions)
    produced = list(executor.execute(action))
    errors = [m for m in produced if type(m) is ErrorMessage]
    assert len(errors) == 1
    if action_kind != "prompt":
        calls = [m for m in produced if isinstance(m, ToolCallMessage)]
        results = [m for m in produced if isinstance(m, ToolResultMessage)]
        assert len(calls) == len(results) == 1
        assert calls[0].id == results[0].id
        assert (
            produced.index(calls[0])
            < produced.index(errors[0])
            < produced.index(results[0])
        )
    reopened = SessionManager(
        storage_manager,
        session_manager.current.configuration,
        root=storage_manager.root,
    )
    reopened.load(session_manager.current.id)
    saved = [m for m in reopened.visible_messages() if type(m) is ErrorMessage]
    assert [m.content for m in saved] == [errors[0].content]
    assert not any(type(m) is ErrorMessage for m in reopened.model_history())
    if action_kind == "silent_shell":
        assert reopened.model_history() == []


@pytest.mark.parametrize("stop", ["retry", "limit", "post_response"])
def test_final_hook_errors_survive_abnormal_agent_stops(
    plugins_root: Path,
    plugin_manager: PluginManager,
    agent: Agent,
    client: ScriptedClient,
    stop: str,
) -> None:
    """Terminal error and turn-end messages survive early return and exhaustion."""
    for name, hook in [("error", "ERROR"), ("end", "TURN_END")]:
        _write_plugin(
            plugins_root,
            f"stop_{stop}_{name}",
            module=f"""
from arancio.core.hooks.base import Hook
from arancio.core.plugins.decorators import hook, plugin
@plugin(name={name!r})
class Broken:
    @hook(Hook.{hook})
    def explode(self, context):
        raise RuntimeError("stop boom")
""",
        )
    assert plugin_manager.load() == []
    agent.max_retries = 1
    if stop == "limit":
        agent.max_turns = 1
        client.turns = [
            [ToolCallMessage(content="missing", id="c1", name="Missing", arguments={})]
        ]
    elif stop == "post_response":
        client.turns = [[AssistantMessage(content="partial"), RuntimeError("provider")]]
    else:
        client.turns = [[RuntimeError("provider")]]
    produced = list(agent(UserMessage(content="hi")))
    failures = [m for m in produced if "stop boom" in str(m.content)]
    assert len(failures) == 2
    assert len(client.requests) == 1


@pytest.mark.parametrize(
    ("folder", "metadata", "tool_name", "expected_name"),
    [
        ("default_tool_name", {}, None, "NamedTools-get_value"),
        (
            "decorator_tool_name",
            {"name": "NamedPlugin"},
            None,
            "NamedPlugin-get_value",
        ),
        (
            "overridden_tool_name",
            {"name": "NamedPlugin"},
            "fetch",
            "NamedPlugin-fetch",
        ),
    ],
    ids=["class-default", "decorator-name", "tool-override"],
)
def test_plugin_tool_names_reach_permissions_hooks_and_provider_payloads(
    plugins_root: Path,
    plugin_manager: PluginManager,
    agent: Agent,
    client: ScriptedClient,
    hook_manager: HookManager,
    folder: str,
    metadata: dict,
    tool_name: str | None,
    expected_name: str,
) -> None:
    """The complete tool name is used throughout the model and execution pipeline."""
    _write_plugin(
        plugins_root,
        folder,
        module=f"""
        from arancio.core.plugins.decorators import plugin, tool


        @plugin(**{metadata!r})
        class NamedTools:
            @tool(
                name={tool_name!r},
                description="Return the plugin name.",
                input_schema={{"properties": {{}}}},
            )
            def get_value(self) -> str:
                return "ok"
        """,
    )
    assert plugin_manager.load() == []
    agent.set_permission_level(PermissionCategory.PLUGIN, PermissionLevel.AUTO)
    called_names = []
    hook_manager.register(
        Hook.BEFORE_TOOL_CALL, lambda context: called_names.append(context["name"])
    )
    call = ToolCallMessage(
        content="get value", id="c1", name=expected_name, arguments={}
    )
    client.turns = [[call], [AssistantMessage(content="done")]]

    produced = list(agent(UserMessage(content="get the plugin name")))

    assert [type(message) for message in produced] == [
        ToolCallMessage,
        ToolResultMessage,
        AssistantMessage,
    ]
    assert produced[1].id == call.id
    assert produced[1].content == "ok"
    assert called_names == [expected_name]
    assert PermissionCategory.for_tool(expected_name) is PermissionCategory.PLUGIN
    payload = LiteLLMPayloadBuilder.build(client.requests[1])
    tool_names = {tool["name"] for tool in payload["tools"]}
    assert expected_name in tool_names
    assert "get_value" not in tool_names
    definition = next(
        tool for tool in payload["tools"] if tool["name"] == expected_name
    )
    assert definition["description"] == "Return the plugin name."
    assert definition["parameters"] == {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    }
    assert [
        item["name"] for item in payload["input"] if item.get("type") == "function_call"
    ] == [expected_name]


def test_two_plugins_can_expose_the_same_method_name(
    plugins_root: Path,
    plugin_manager: PluginManager,
    agent: Agent,
    client: ScriptedClient,
) -> None:
    """Different plugin prefixes keep both tools callable in the agent catalog."""
    for plugin_name in ("Alpha", "Beta"):
        _write_plugin(
            plugins_root,
            f"shared_method_{plugin_name.lower()}",
            module=f"""
            from arancio.core.plugins.decorators import plugin, tool


            @plugin(name={plugin_name!r})
            class SharedTools:
                @tool(
                    description="Return the plugin name.",
                    input_schema={{"properties": {{}}}},
                )
                def echo(self) -> str:
                    return {plugin_name!r}
            """,
        )
    assert plugin_manager.load() == []
    agent.set_permission_level(PermissionCategory.PLUGIN, PermissionLevel.AUTO)
    client.turns = [
        [
            ToolCallMessage(content="first", id="c1", name="Alpha-echo", arguments={}),
            ToolCallMessage(content="second", id="c2", name="Beta-echo", arguments={}),
        ],
        [AssistantMessage(content="done")],
    ]

    produced = list(agent(UserMessage(content="call both plugins")))

    results = [
        message for message in produced if isinstance(message, ToolResultMessage)
    ]
    assert [type(result) for result in results] == [
        ToolResultMessage,
        ToolResultMessage,
    ]
    assert [(result.id, result.content) for result in results] == [
        ("c1", "Alpha"),
        ("c2", "Beta"),
    ]
    assert produced[-1] == AssistantMessage(content="done")


def test_a_tool_with_no_category_lands_in_the_plugin_category(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """The decorator's default category is PLUGIN."""
    _write_plugin(
        plugins_root,
        "defaulter",
        module="""
        from arancio.core.plugins.decorators import plugin, tool


        @plugin()
        class Defaulter:
            @tool(description="Does nothing.", input_schema={"properties": {}})
            def noop(self) -> None:
                pass
        """,
    )

    assert plugin_manager.load() == []

    names = {tool_cls.__name__ for tool_cls in PermissionCategory.PLUGIN.tools}
    assert "Defaulter-noop" in names


def test_a_tool_can_join_an_existing_built_in_category(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """A ``@tool(category="READ")`` method joins READ alongside ReadFileTool."""
    _write_plugin(
        plugins_root,
        "extra_reader",
        module="""
        from arancio.core.plugins.decorators import plugin, tool


        @plugin()
        class ExtraReader:
            @tool(
                description="Reads something else.",
                input_schema={"properties": {}},
                category="READ",
            )
            def read_extra(self) -> None:
                pass
        """,
    )

    assert plugin_manager.load() == []

    names = {tool_cls.__name__ for tool_cls in PermissionCategory.READ.tools}
    assert "ReadFileTool" in names
    assert "ExtraReader-read_extra" in names


def test_custom_tools_on_one_plugin_share_one_dynamic_category(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """CUSTOM tools on one plugin use its decorator name for the category."""
    _write_plugin(
        plugins_root,
        "custom_plugin",
        module="""
        from arancio.core.plugins.decorators import plugin, tool


        @plugin(name="CustomTools")
        class TestPlugin:
            @tool(
                description="First.",
                input_schema={"properties": {}},
                category="CUSTOM",
            )
            def first(self) -> None:
                pass

            @tool(
                description="Second.",
                input_schema={"properties": {}},
                category="CUSTOM",
            )
            def second(self) -> None:
                pass
        """,
    )

    assert plugin_manager.load() == []

    category = PermissionCategory.get("PLUGIN:CustomTools")
    assert category is not None
    assert {tool_cls.__name__ for tool_cls in category.tools} == {
        "CustomTools-first",
        "CustomTools-second",
    }


def test_an_unknown_category_is_an_error_and_skips_only_that_tool(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """One bad category costs one message and does not sink the whole plugin."""
    _write_plugin(
        plugins_root,
        "half_broken",
        module="""
        from arancio.core.plugins.decorators import plugin, tool


        @plugin()
        class HalfBroken:
            @tool(description="Fine.", input_schema={"properties": {}})
            def fine(self) -> None:
                pass

            @tool(
                description="Broken.",
                input_schema={"properties": {}},
                category="BOGUS",
            )
            def broken(self) -> None:
                pass
        """,
    )

    messages = plugin_manager.load()

    assert len(messages) == 1
    assert isinstance(messages[0], ErrorMessage)
    assert "'broken'" in messages[0].content
    assert "unknown permission category" in messages[0].content
    loaded_names = [
        type(plugin.instance).__name__ for plugin in plugin_manager.registry
    ]
    assert loaded_names == ["HalfBroken"]
    names = {tool_cls.__name__ for tool_cls in PermissionCategory.PLUGIN.tools}
    assert "HalfBroken-fine" in names
    all_tool_names = {
        tool_cls.__name__
        for category in PermissionCategory
        for tool_cls in category.tools
    }
    assert "HalfBroken-broken" not in all_tool_names


def test_a_plugin_tool_is_callable_end_to_end(
    plugins_root: Path,
    plugin_manager: PluginManager,
    tool_manager: ToolManager,
    controller: ScriptedController,
    hook_manager: HookManager,
) -> None:
    """A @tool method reaches the bound plugin instance and returns its result."""
    _write_plugin(
        plugins_root,
        "doubler",
        module="""
        from arancio.core.plugins.decorators import plugin, tool


        @plugin()
        class Doubler:
            @tool(
                description="Doubles a number.",
                input_schema={
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            )
            def double(self, x: int) -> int:
                return x * 2
        """,
    )

    assert plugin_manager.load() == []

    # constructed only after plugin_manager.load(), so the default grants it
    # builds already cover the newly registered PLUGIN-category tool
    permission_manager = PermissionManager(
        tool_manager=tool_manager, controller=controller, hook_manager=hook_manager
    )
    tools = {tool.name: tool for tool in permission_manager.allowed_tools}

    result, hook_messages = tools["Doubler-double"].call(call_id="c1", x=21)

    assert hook_messages == []
    assert isinstance(result, ToolResultMessage)
    assert result.content == 42


@pytest.mark.parametrize(
    ("name", "output"),
    [
        ("empty_result", {}),
        (
            "nested_result",
            {
                "ok": True,
                "items": [{"name": "caffè", "count": 2}],
                "note": None,
            },
        ),
    ],
    ids=["empty", "nested"],
)
def test_a_plugin_dictionary_result_reaches_hooks_and_the_next_turn(
    plugins_root: Path,
    plugin_manager: PluginManager,
    agent: Agent,
    client: ScriptedClient,
    name: str,
    output: dict,
) -> None:
    """Dictionary results stay structured and do not stop the agent run."""
    _write_plugin(
        plugins_root,
        name,
        module="""
        from arancio.core.hooks.base import Hook
        from arancio.core.plugins.decorators import hook, plugin, tool

        seen = []


        @plugin()
        class Reporter:
            @tool(
                description="Returns the supplied data.",
                input_schema={
                    "properties": {"payload": {"type": "object"}},
                    "required": ["payload"],
                },
            )
            def report(self, payload: dict) -> dict:
                return payload

            @hook(Hook.AFTER_TOOL_CALL)
            def record_result(self, context) -> None:
                seen.append(context["result"])
        """,
    )
    assert plugin_manager.load() == []
    agent.set_permission_level(PermissionCategory.PLUGIN, PermissionLevel.AUTO)
    call = ToolCallMessage(
        content="report", id="c1", name="Reporter-report", arguments={"payload": output}
    )
    reply = AssistantMessage(content="done")
    client.turns = [[call], [reply]]

    produced = list(agent(UserMessage(content="report the data")))

    assert [type(message) for message in produced] == [
        ToolCallMessage,
        ToolResultMessage,
        AssistantMessage,
    ]
    result = produced[1]
    assert result.id == call.id
    assert result.content is output
    assert json.loads(result.display_text) == output
    assert result.in_history
    assert _loaded_module(name).seen == [result]
    assert len(client.histories) == 2
    assert client.histories[1][-1] is result
    assert produced[-1] is reply


def test_plugin_decorator_preserves_the_original_class() -> None:
    """Decoration leaves the original class and its attributes unchanged."""

    class Plain:
        """A class whose constructor must not run during decoration."""

        def __init__(self) -> None:
            """Reject construction to detect an eager decorator.

            Raises:
                RuntimeError: if the decorator constructs the class.
            """
            raise RuntimeError("constructed")

    attributes = dict(vars(Plain))
    assert plugin(name="Marked")(Plain) is Plain
    assert dict(vars(Plain)) == attributes
    assert Plain.__bases__ == (object,)


def test_plugin_code_has_no_direct_metadata_access(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """Plugin objects and methods expose no decorator metadata at any call site."""
    from arancio.core.plugins.registry import PLUGIN_REGISTRY

    _write_plugin(
        plugins_root,
        "external_metadata",
        module="""
from arancio.core.hooks.base import Hook
from arancio.core.plugins.decorators import hook, plugin, tool

checked = []

@plugin(name="PrivateMetadata", version="1.2.3")
class Probe:
    def __init__(self):
        self.check_metadata()

    def check_metadata(self):
        for owner in (type(self), self):
            for field in (
                "__plugin_metadata", "__arancio_plugin__", "metadata",
                "manifest", "name", "source_path", "instance",
            ):
                assert not hasattr(owner, field), field
        for method in (
            type(self).on_start, self.on_start, type(self).probe, self.probe,
        ):
            assert not hasattr(method, "__arancio_hooks__")
            assert not hasattr(method, "__arancio_tool__")
        checked.append(self)
        return "no metadata"

    @hook(Hook.STARTUP)
    def on_start(self, context):
        self.check_metadata()

    @tool(description="Check metadata access", input_schema={})
    def probe(self):
        return self.check_metadata()
""",
    )

    assert plugin_manager.load() == []
    (loaded,) = plugin_manager.registry
    assert loaded.name == "PrivateMetadata"
    assert loaded.version == "1.2.3"
    assert loaded.description == ""
    assert loaded.author == ""
    assert loaded.enabled is True
    assert not hasattr(loaded, "metadata")
    assert plugin_manager.registry is PLUGIN_REGISTRY
    assert isinstance(loaded, Plugin)
    assert hook_manager.run(Hook.STARTUP) == []

    (tool_class,) = PermissionCategory.PLUGIN.tools
    result, messages = tool_class(hook_manager=hook_manager).call(call_id="c1")
    assert tool_class.__name__ == "PrivateMetadata-probe"
    assert result.content == "no metadata"
    assert messages == []
    assert _loaded_module("external_metadata").checked == [loaded.instance] * 3


@pytest.mark.parametrize("tool_outermost", [True, False])
def test_method_decorators_preserve_the_original_function(tool_outermost: bool) -> None:
    """Both decorator orders keep hooks and tools without changing the function."""
    from arancio.core.plugins.decorators import hook, tool

    class Plain:
        """A class decorated only after its method declarations are collected."""

        def probe(self) -> None:
            """Provide a plain function for the decorator checks."""

    original = Plain.probe
    attributes = dict(vars(original))
    decorated = hook(Hook.TURN_END)(original)
    tool_decorator = tool(description="Probe", input_schema={})
    if tool_outermost:
        decorated = tool_decorator(hook(Hook.TURN_START)(decorated))
    else:
        decorated = hook(Hook.TURN_START)(tool_decorator(decorated))

    assert decorated is original
    assert dict(vars(original)) == attributes
    assert plugin()(Plain) is Plain
    (record,) = PLUGIN_REGISTRY
    assert record.hooks == [(original, (Hook.TURN_END, Hook.TURN_START))]
    (tool_class,) = record.tools
    assert issubclass(tool_class, PluginTool)
    assert tool_class._method is original
    assert tool_class.__name__ == "Plain-probe"
    assert tool_class.tool_name == "probe"
    assert tool_class._description == "Probe"
    assert tool_class._input_schema == {}
    assert tool_class.category == "PLUGIN"


def test_shared_method_declarations_are_available_to_each_plugin(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """Collecting one class cannot consume a shared function's hooks or tool fields."""
    _write_plugin(
        plugins_root,
        "shared_declarations",
        module="""
from arancio.core.hooks.base import Hook
from arancio.core.plugins.decorators import hook, plugin, tool

seen = []

@hook(Hook.STARTUP)
@tool(description="Shared", input_schema={})
def shared(self, context=None):
    seen.append(type(self).__name__)
    return type(self).__name__

@plugin()
class Alpha:
    greet = shared

@plugin()
class Beta:
    greet = shared
""",
    )

    assert plugin_manager.load() == []
    assert [item.name for item in plugin_manager.registry] == [
        "Alpha",
        "Beta",
    ]
    assert hook_manager.run(Hook.STARTUP) == []
    tool_classes = {item.__name__: item for item in PermissionCategory.PLUGIN.tools}
    for plugin_name in ("Alpha", "Beta"):
        tool_class = tool_classes[f"{plugin_name}-shared"]
        result, messages = tool_class(hook_manager=hook_manager).call(call_id="c1")
        assert result.content == plugin_name
        assert messages == []
    assert _loaded_module("shared_declarations").seen == [
        "Alpha",
        "Beta",
        "Alpha",
        "Beta",
    ]


def test_registry_collection_ignores_undecorated_callable_fields(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """An unhashable callable field does not prevent real methods from loading."""
    _write_plugin(
        plugins_root,
        "callable_configuration",
        module="""
from arancio.core.hooks.base import Hook
from arancio.core.plugins.decorators import hook, plugin

class Configuration:
    __hash__ = None

    def __call__(self):
        return "configured"

@plugin()
class Configured:
    configuration = Configuration()

    @hook(Hook.STARTUP)
    def start(self, context):
        context["configuration"] = self.configuration()
""",
    )

    assert plugin_manager.load() == []
    assert hook_manager.run(Hook.STARTUP) == []
    assert hook_manager.context["configuration"] == "configured"


@pytest.mark.parametrize(
    "same_file", [True, False], ids=["same-file", "different-files"]
)
@pytest.mark.parametrize("second_name", ["Repeated", "repeated"], ids=["exact", "case"])
def test_all_conflicting_plugins_are_skipped_before_construction_and_registration(
    plugins_root: Path,
    plugin_manager: PluginManager,
    hook_manager: HookManager,
    same_file: bool,
    second_name: str,
) -> None:
    """No conflicting class can construct, register hooks or create a category."""
    modules = []
    for class_name, plugin_name in (("First", "Repeated"), ("Second", second_name)):
        modules.append(f"""
from arancio.core.hooks.base import Hook
from arancio.core.plugins.decorators import hook, plugin, tool

@plugin(name={plugin_name!r})
class {class_name}:
    def __init__(self):
        raise RuntimeError("conflicting constructor ran")

    @hook(Hook.STARTUP)
    def start(self, context):
        raise RuntimeError("conflicting hook ran")

    @tool(description="Conflict", input_schema={{}}, category="CUSTOM")
    def conflict(self):
        return "conflict"
""")
    if same_file:
        _write_plugin(plugins_root, "aaa_conflicts", module="\n".join(modules))
    else:
        for index, module in enumerate(modules):
            _write_plugin(plugins_root, f"aaa_conflict_{index}", module=module)
    _write_plugin(plugins_root, "zzz_healthy")

    messages = plugin_manager.load()

    assert len(messages) == 2
    assert all(isinstance(message, ErrorMessage) for message in messages)
    assert all("duplicate plugin name" in message.content for message in messages)
    assert all("plugin.py:" in message.content for message in messages)
    assert [item.name for item in plugin_manager.registry] == [
        "Repeated",
        second_name,
        "Logger",
    ]
    assert all(item.instance is None for item in plugin_manager.registry[:2])
    assert plugin_manager.registry[-1].instance is not None
    assert PermissionCategory.get("PLUGIN:Repeated") is None
    assert hook_manager.run(Hook.STARTUP) == []


def test_disabled_classes_do_not_conflict_with_enabled_plugins(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """A disabled class can share a name without blocking an enabled class."""
    module = _LOGGER_MODULE + _SECOND_CLASS.replace(
        "@plugin()", "@plugin(name='Logger', enabled=False)"
    )
    _write_plugin(plugins_root, "disabled_duplicate", module=module)

    assert plugin_manager.load() == []
    assert [item.name for item in plugin_manager.registry] == ["Logger", "Logger"]
    assert plugin_manager.registry[0].instance is not None
    assert plugin_manager.registry[1].instance is None


@pytest.mark.parametrize(
    "constructor",
    [
        "def __init__(self, required): pass",
        "def __init__(self): raise RuntimeError('constructor boom')",
    ],
    ids=["required-argument", "raises"],
)
def test_constructor_failures_skip_only_the_failing_class(
    plugins_root: Path, plugin_manager: PluginManager, constructor: str
) -> None:
    """A constructor error leaves sibling plugins in the same file available."""
    module = (
        _LOGGER_MODULE
        + f"""
@plugin()
class Broken:
    {constructor}

    @hook(Hook.STARTUP)
    def start(self, context): pass
"""
    )
    _write_plugin(plugins_root, "constructor_failure", module=module)

    messages = plugin_manager.load()

    assert len(messages) == 1
    assert isinstance(messages[0], ErrorMessage)
    assert "plugin.py:Broken" in messages[0].content
    assert "cannot be constructed" in messages[0].content
    assert [item.name for item in plugin_manager.registry] == ["Logger", "Broken"]
    assert plugin_manager.registry[0].instance is not None
    assert plugin_manager.registry[1].instance is None


def test_decorators_register_helpers_but_not_aliases_or_undecorated_subclasses(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """Every decorated class registers once, including imported helper classes."""
    module = (
        _LOGGER_MODULE
        + """
from .helpers import Imported
Alias = Logger
class Child(Logger): pass
"""
    )
    _write_plugin(
        plugins_root,
        "class_selection",
        module=module,
        extra_files={"helpers.py": _LOGGER_MODULE.replace("Logger", "Imported")},
    )

    assert plugin_manager.load() == []
    assert [item.name for item in plugin_manager.registry] == ["Logger", "Imported"]


def test_an_empty_decorated_subclass_loads_with_a_warning(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """A decorated subclass with no own declarations loads and reports a warning."""
    module = (
        _LOGGER_MODULE
        + """
@plugin()
class Child(Logger): pass
"""
    )
    _write_plugin(plugins_root, "inherited_handlers", module=module)

    messages = plugin_manager.load()

    assert len(messages) == 1
    assert isinstance(messages[0], WarningMessage)
    assert "declares no hooks and no tools" in messages[0].content
    assert [item.name for item in plugin_manager.registry] == ["Logger", "Child"]
    assert plugin_manager.registry[1].instance is not None
    assert plugin_manager.registry[1].hooks == []


def test_hooks_and_tools_share_only_their_own_plugin_instance(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """Independent classes keep their state when tool instances are rebuilt."""
    module = """
from arancio.core.hooks.base import Hook
from arancio.core.plugins.decorators import hook, plugin, tool
constructed = []
"""
    for class_name, start in (("Alpha", 0), ("Beta", 10)):
        module += f"""
@plugin()
class {class_name}:
    def __init__(self):
        constructed.append({class_name!r})
        self.count = {start}

    @hook(Hook.STARTUP)
    def start(self, context):
        self.count += 1

    @tool(description="Get count", input_schema={{}})
    def count_value(self):
        return self.count
"""
    _write_plugin(plugins_root, "independent_state", module=module)
    assert plugin_manager.load() == []

    hook_manager.run(Hook.STARTUP)
    classes = {item.__name__: item for item in PermissionCategory.PLUGIN.tools}
    for plugin_name, expected in (("Alpha", 1), ("Beta", 11)):
        tool_class = classes[f"{plugin_name}-count_value"]
        for _ in range(2):
            result, messages = tool_class(hook_manager=hook_manager).call(call_id="c1")
            assert result.content == expected
            assert messages == []
    assert _loaded_module("independent_state").constructed == ["Alpha", "Beta"]


def test_disabling_a_shared_hook_function_is_scoped_to_its_plugin_instance(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """One plugin's failed method cannot disable another class using that function."""
    module = """
from arancio.core.hooks.base import Hook
from arancio.core.plugins.decorators import hook, plugin

@hook(Hook.STARTUP)
def shared(self, context):
    self.calls += 1
    if type(self).__name__ == "Broken":
        raise RuntimeError("shared hook failed")

@plugin()
class Broken:
    __hash__ = None
    def __init__(self): self.calls = 0
    start = shared

@plugin()
class Healthy:
    __hash__ = None
    def __init__(self): self.calls = 0
    start = shared
"""
    _write_plugin(plugins_root, "shared_hook_function", module=module)
    assert plugin_manager.load() == []

    assert len(hook_manager.run(Hook.STARTUP)) == 1
    assert hook_manager.run(Hook.STARTUP) == []
    assert [item.instance.calls for item in plugin_manager.registry] == [1, 2]


def test_relative_helpers_and_custom_permissions_are_separate_for_matching_class_names(
    plugins_root: Path,
    plugin_manager: PluginManager,
    tool_manager: ToolManager,
    controller: ScriptedController,
    hook_manager: HookManager,
) -> None:
    """Names separate permissions and import packages separate local helpers."""
    for plugin_name in ("Alpha", "Beta"):
        _write_plugin(
            plugins_root,
            f"helper_{plugin_name.lower()}",
            module=f"""
from arancio.core.plugins.decorators import plugin, tool
from .helpers import VALUE

@plugin(name={plugin_name!r})
class SameClass:
    @tool(description="Local value", input_schema={{}}, category="CUSTOM")
    def value(self): return VALUE
""",
            extra_files={"helpers.py": f"VALUE = {plugin_name!r}\n"},
        )
    assert plugin_manager.load() == []
    permissions = PermissionManager(
        tool_manager=tool_manager, controller=controller, hook_manager=hook_manager
    )
    for plugin_name in ("Alpha", "Beta"):
        category = PermissionCategory.get(f"PLUGIN:{plugin_name}")
        assert category is not None
        permissions.validate(
            ToolCallMessage(
                content="local value",
                id="c1",
                name=f"{plugin_name}-value",
                arguments={},
            )
        )
        (tool_class,) = category.tools
        assert tool_class.__name__ == f"{plugin_name}-value"
        result, messages = tool_class(hook_manager=hook_manager).call(call_id="c1")
        assert result.content == plugin_name
        assert messages == []
    assert PermissionCategory.get("PLUGIN:SameClass") is None
    assert len(controller.requests) == 2

    settings, messages = SettingsValidator.validate(
        {
            "provider": "openai",
            "model_name": "gpt-4o",
            "permissions": {"PLUGIN:SameClass": "auto", "PLUGIN:Alpha": "auto"},
        }
    )
    assert (
        settings.permissions[PermissionCategory["PLUGIN:Alpha"]] is PermissionLevel.AUTO
    )
    assert (
        settings.permissions[PermissionCategory["PLUGIN:Beta"]] is PermissionLevel.ASK
    )
    assert any(
        isinstance(message, WarningMessage) and "PLUGIN:SameClass" in message.content
        for message in messages
    )
