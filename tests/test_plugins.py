"""Tests for plugin discovery, loading, hook attachment and failure isolation."""

import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from fakes import RecordingApp, ScriptedClient, ScriptedController

from arancio.core.agents import Agent
from arancio.core.hooks.manager import HookManager
from arancio.core.hooks.types import Hook
from arancio.core.messages import (
    AssistantMessage,
    ErrorMessage,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
    WarningMessage,
)
from arancio.core.permissions.manager import PermissionManager
from arancio.core.permissions.types import PermissionCategory
from arancio.core.plugins.manager import PluginManager
from arancio.core.plugins.manifest import PluginManifestValidator
from arancio.core.tools.manager import ToolManager
from arancio.prompt.actions.executor import ActionExecutor
from arancio.prompt.actions.types import PromptAction, ShellCommandAction
from arancio.sessions.manager import SessionManager
from arancio.settings.manager import SettingsManager
from arancio.storage.manager import StorageManager

_LOGGER_MODULE = """
from arancio.core.hooks.types import Hook
from arancio.core.plugins.base import Plugin
from arancio.core.plugins.hook import hook

calls = []


class Logger(Plugin):
    @hook(Hook.AGENT_START)
    def on_agent_start(self, **kwargs) -> None:
        calls.append(kwargs)
"""

_SECOND_CLASS = """

class Second(Plugin):
    @hook(Hook.AGENT_END)
    def on_agent_end(self, **kwargs) -> None:
        pass
"""


def _write_plugin(
    root: Path,
    name: str,
    manifest: str | None = "",
    module: str | None = _LOGGER_MODULE,
    extra_files: dict[str, str] | None = None,
) -> Path:
    """Create one plugin folder on disk.

    Args:
        root: the plugins root the folder is created under.
        name: the plugin's folder name.
        manifest: the ``manifest.yml`` body, or ``None`` to leave it out.
        module: the ``module.py`` body, or ``None`` to leave it out.
        extra_files: further files to drop in the folder, keyed by filename.

    Returns:
        The created plugin folder.
    """
    directory = root / name
    directory.mkdir(parents=True)
    if manifest is not None:
        (directory / "manifest.yml").write_text(manifest)
    if module is not None:
        (directory / "module.py").write_text(textwrap.dedent(module))
    for filename, content in (extra_files or {}).items():
        (directory / filename).write_text(textwrap.dedent(content))
    return directory


def _loaded_module(name: str) -> Any:
    """Return a loaded plugin's module, to read what its plugin recorded.

    Args:
        name: the plugin's folder name.

    Returns:
        The module object the loader put in ``sys.modules``.
    """
    return sys.modules[f"arancio_plugins.{name}.module"]


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


def test_an_empty_manifest_is_valid() -> None:
    """Every manifest field has a default, so an empty mapping validates."""
    manifest, messages = PluginManifestValidator.validate({}, "my_plugin", "loc")

    assert messages == []
    assert manifest.name == "my_plugin"
    assert manifest.version == "0.0.0"
    assert manifest.enabled is True


def test_a_manifest_names_the_plugin_over_its_folder() -> None:
    """An explicit name wins over the folder name it defaults to."""
    manifest, messages = PluginManifestValidator.validate(
        {"name": "Tool call logger"}, "tool_call_logger", "loc"
    )

    assert messages == []
    assert manifest.name == "Tool call logger"


def test_an_invalid_manifest_field_falls_back_and_warns() -> None:
    """A wrongly typed field takes its default and produces one warning."""
    manifest, messages = PluginManifestValidator.validate(
        {"enabled": "yes"}, "my_plugin", "plugins/my_plugin/manifest.yml"
    )

    assert manifest.enabled is True
    assert len(messages) == 1
    assert isinstance(messages[0], WarningMessage)
    assert "plugins/my_plugin/manifest.yml" in messages[0].content
    assert "enabled" in messages[0].content


def test_an_unknown_manifest_field_is_dropped_with_a_warning() -> None:
    """A key naming no known field is ignored and reported."""
    _, messages = PluginManifestValidator.validate({"bogus": 1}, "my_plugin", "loc")

    assert len(messages) == 1
    assert isinstance(messages[0], WarningMessage)
    assert "bogus" in messages[0].content


def test_a_manifest_that_is_not_a_mapping_is_an_error(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """A manifest holding a scalar cannot describe a plugin."""
    _write_plugin(plugins_root, "scalar_manifest", manifest="just a string")

    messages = plugin_manager.load()

    assert plugin_manager.plugins == []
    assert len(messages) == 1
    assert isinstance(messages[0], ErrorMessage)
    assert "must be a mapping" in messages[0].content


def test_unparseable_yaml_in_the_manifest_is_an_error(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """A syntactically broken manifest is reported, not raised."""
    _write_plugin(plugins_root, "broken_yaml", manifest="name: [unclosed")

    messages = plugin_manager.load()

    assert plugin_manager.plugins == []
    assert len(messages) == 1
    assert isinstance(messages[0], ErrorMessage)
    assert "cannot be read" in messages[0].content


def test_a_folder_without_a_manifest_is_not_a_plugin(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """The manifest is the marker, so a folder without one is skipped silently."""
    _write_plugin(plugins_root, "no_manifest", manifest=None)

    assert plugin_manager.load() == []
    assert plugin_manager.plugins == []


def test_underscore_and_dot_prefixed_folders_are_skipped(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """``__pycache__`` and hidden folders never count as plugins."""
    _write_plugin(plugins_root, "__pycache__")
    _write_plugin(plugins_root, ".hidden")

    assert plugin_manager.load() == []
    assert plugin_manager.plugins == []


def test_a_missing_plugins_root_reports_nothing(
    tmp_path: Path, plugin_manager: PluginManager
) -> None:
    """Nothing creates the directory, so not having one just means no plugins."""
    assert not (tmp_path / "plugins").exists()

    assert plugin_manager.load() == []
    assert plugin_manager.plugins == []


def test_a_missing_plugin_module_is_an_error(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """A manifest with no ``module.py`` beside it is a broken plugin."""
    _write_plugin(plugins_root, "no_module", module=None)

    messages = plugin_manager.load()

    assert plugin_manager.plugins == []
    assert len(messages) == 1
    assert isinstance(messages[0], ErrorMessage)
    assert "module.py not found" in messages[0].content


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
    assert [plugin.directory.name for plugin in plugin_manager.plugins] == [
        "zzz_healthy"
    ]


def test_a_module_with_no_plugin_class_is_an_error(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """A module importing a base but defining nothing is not a plugin.

    The imported base must not be mistaken for the plugin itself.
    """
    _write_plugin(
        plugins_root,
        "empty_module",
        module="from arancio.core.plugins.base import Plugin\n",
    )

    messages = plugin_manager.load()

    assert plugin_manager.plugins == []
    assert len(messages) == 1
    assert isinstance(messages[0], ErrorMessage)
    assert "defines no plugin class" in messages[0].content


def test_a_module_with_two_plugin_classes_is_an_error(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """One folder is one plugin, so an ambiguous module is refused."""
    _write_plugin(plugins_root, "two_classes", module=_LOGGER_MODULE + _SECOND_CLASS)

    messages = plugin_manager.load()

    assert plugin_manager.plugins == []
    assert len(messages) == 1
    assert isinstance(messages[0], ErrorMessage)
    assert "more than one plugin class" in messages[0].content
    assert "Logger, Second" in messages[0].content


def test_a_plugin_with_no_hooks_and_no_tools_is_an_error(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """A plugin binding nothing and defining no tools could never run."""
    _write_plugin(
        plugins_root,
        "no_hooks",
        module="""
        from arancio.core.plugins.base import Plugin


        class Idle(Plugin):
            def not_a_hook(self, **kwargs) -> None:
                pass
        """,
    )

    messages = plugin_manager.load()

    assert plugin_manager.plugins == []
    assert len(messages) == 1
    assert isinstance(messages[0], ErrorMessage)
    assert "declares no hooks and no tools" in messages[0].content


def test_a_tools_only_plugin_with_no_hooks_loads_successfully(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """A plugin defining only a @tool method no longer needs any hooks."""
    _write_plugin(
        plugins_root,
        "tools_only",
        module="""
        from arancio.core.plugins.base import Plugin
        from arancio.core.plugins.tool import tool


        class Idle(Plugin):
            @tool(description="Does nothing.", input_schema={"properties": {}})
            def noop(self) -> None:
                pass
        """,
    )

    messages = plugin_manager.load()

    assert messages == []
    assert [type(plugin).__name__ for plugin in plugin_manager.plugins] == ["Idle"]


def test_a_disabled_plugin_loads_nothing_and_reports_nothing(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """``enabled: false`` is a deliberate choice, not a problem to report."""
    _write_plugin(plugins_root, "switched_off", manifest="enabled: false")

    assert plugin_manager.load() == []
    assert plugin_manager.plugins == []

    hook_manager.run(Hook.AGENT_START, message=UserMessage(content="hi"))


def test_a_plugin_can_import_a_sibling_module_from_its_own_folder(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """The synthetic package makes a plugin's own modules reachable."""
    _write_plugin(
        plugins_root,
        "with_helpers",
        module="""
        from arancio.core.hooks.types import Hook
        from arancio.core.plugins.base import Plugin
        from arancio.core.plugins.hook import hook

        from .helpers import GREETING


        class Greeter(Plugin):
            @hook(Hook.AGENT_START)
            def greet(self, **kwargs) -> None:
                (self.directory / "out.txt").write_text(GREETING)
        """,
        extra_files={"helpers.py": "GREETING = 'hello'\n"},
    )

    assert plugin_manager.load() == []

    hook_manager.run(Hook.AGENT_START, message=UserMessage(content="hi"))

    assert (plugins_root / "with_helpers" / "out.txt").read_text() == "hello"


def test_a_registered_plugin_runs_on_dispatch_with_the_hook_kwargs(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """The hook method receives that hook's own kwargs."""
    directory = _write_plugin(plugins_root, "records_calls")

    assert plugin_manager.load() == []
    assert [plugin.name for plugin in plugin_manager.plugins] == ["records_calls"]
    assert plugin_manager.plugins[0].directory == directory

    message = UserMessage(content="hi")
    hook_manager.run(Hook.AGENT_START, message=message)

    assert _loaded_module("records_calls").calls == [{"message": message}]


def test_two_hook_methods_each_run_only_on_their_own_hook(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """One plugin can bind a different method to each hook."""
    _write_plugin(
        plugins_root,
        "two_methods",
        module="""
        from arancio.core.hooks.types import Hook
        from arancio.core.plugins.base import Plugin
        from arancio.core.plugins.hook import hook

        seen = []


        class Split(Plugin):
            @hook(Hook.TURN_START)
            def on_start(self, turn) -> None:
                seen.append(("start", turn))

            @hook(Hook.TURN_END)
            def on_end(self, turn) -> None:
                seen.append(("end", turn))
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
        from arancio.core.hooks.types import Hook
        from arancio.core.plugins.base import Plugin
        from arancio.core.plugins.hook import hook

        seen = []


        class Both(Plugin):
            @hook(Hook.TURN_START)
            @hook(Hook.TURN_END)
            def on_turn(self, turn) -> None:
                seen.append(turn)
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
        from arancio.core.hooks.types import Hook
        from arancio.core.plugins.base import Plugin
        from arancio.core.plugins.hook import hook


        class Broken(Plugin):
            @hook(Hook.AGENT_START)
            def on_agent_start(self, **kwargs) -> None:
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
        from arancio.core.hooks.types import Hook
        from arancio.core.plugins.base import Plugin
        from arancio.core.plugins.hook import hook

        runs = []


        class Broken(Plugin):
            @hook(Hook.AGENT_START)
            def explode(self, **kwargs) -> None:
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
    assert "always_raises" in notices[0].content
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
        from arancio.core.hooks.types import Hook
        from arancio.core.plugins.base import Plugin
        from arancio.core.plugins.hook import hook

        broken_runs = []
        healthy_runs = []


        class Mixed(Plugin):
            @hook(Hook.TURN_START)
            @hook(Hook.TURN_END)
            def broken(self, turn) -> None:
                broken_runs.append(turn)
                raise RuntimeError("boom")

            @hook(Hook.TURN_END)
            def healthy(self, turn) -> None:
                healthy_runs.append(turn)
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
        from arancio.core.hooks.types import Hook
        from arancio.core.plugins.base import Plugin
        from arancio.core.plugins.hook import hook


        class Broken(Plugin):
            @hook(Hook.TURN_START)
            def on_turn_start(self, **kwargs) -> None:
                raise RuntimeError("boom")
        """,
    )
    assert plugin_manager.load() == []

    produced = list(agent(message=UserMessage(content="hi")))

    assert [type(message) for message in produced] == [ErrorMessage, AssistantMessage]


@pytest.mark.parametrize("hook", list(Hook))
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
from arancio.core.hooks.types import Hook
from arancio.core.plugins.base import Plugin
from arancio.core.plugins.hook import hook
class Broken(Plugin):
    @hook(Hook.{hook.name})
    def explode(self, **kwargs):
        raise RuntimeError("plugin boom")
""",
    )
    plugin_manager.load()
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
from arancio.core.hooks.types import Hook
from arancio.core.plugins.base import Plugin
from arancio.core.plugins.hook import hook
runs = []
class Broken(Plugin):
    @hook(Hook.TURN_END)
    def explode(self, **kwargs):
        runs.append(1)
        raise RuntimeError("close boom")
""",
    )
    plugin_manager.load()
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
from arancio.core.hooks.types import Hook
from arancio.core.plugins.base import Plugin
from arancio.core.plugins.hook import hook
class Broken(Plugin):
    @hook(Hook.{hook})
    def explode(self, **kwargs):
        raise RuntimeError("persist boom")
""",
    )
    plugin_manager.load()
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
from arancio.core.hooks.types import Hook
from arancio.core.plugins.base import Plugin
from arancio.core.plugins.hook import hook
class Broken(Plugin):
    @hook(Hook.{hook})
    def explode(self, **kwargs):
        raise RuntimeError("stop boom")
""",
        )
    plugin_manager.load()
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


def test_a_tool_with_no_category_lands_in_the_plugin_category(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """The decorator's default category is PLUGIN."""
    _write_plugin(
        plugins_root,
        "defaulter",
        module="""
        from arancio.core.plugins.base import Plugin
        from arancio.core.plugins.tool import tool


        class Defaulter(Plugin):
            @tool(description="Does nothing.", input_schema={"properties": {}})
            def noop(self) -> None:
                pass
        """,
    )

    assert plugin_manager.load() == []

    assert "noop" in {tool_cls.__name__ for tool_cls in PermissionCategory.PLUGIN.tools}


def test_a_tool_can_join_an_existing_built_in_category(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """A ``@tool(category="READ")`` method joins READ alongside ReadFileTool."""
    _write_plugin(
        plugins_root,
        "extra_reader",
        module="""
        from arancio.core.plugins.base import Plugin
        from arancio.core.plugins.tool import tool


        class ExtraReader(Plugin):
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
    assert "read_extra" in names


def test_custom_tools_on_one_plugin_share_one_dynamic_category(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """CUSTOM tools on the same plugin land in one PLUGIN:<ClassName> category."""
    _write_plugin(
        plugins_root,
        "custom_plugin",
        module="""
        from arancio.core.plugins.base import Plugin
        from arancio.core.plugins.tool import tool


        class TestPlugin(Plugin):
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

    category = PermissionCategory.get("PLUGIN:TestPlugin")
    assert category is not None
    assert {tool_cls.__name__ for tool_cls in category.tools} == {"first", "second"}


def test_an_unknown_category_is_an_error_and_skips_only_that_tool(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """One bad category costs one message and does not sink the whole plugin."""
    _write_plugin(
        plugins_root,
        "half_broken",
        module="""
        from arancio.core.plugins.base import Plugin
        from arancio.core.plugins.tool import tool


        class HalfBroken(Plugin):
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
    loaded_names = [type(plugin).__name__ for plugin in plugin_manager.plugins]
    assert loaded_names == ["HalfBroken"]
    assert "fine" in {tool_cls.__name__ for tool_cls in PermissionCategory.PLUGIN.tools}
    all_tool_names = {
        tool_cls.__name__
        for category in PermissionCategory
        for tool_cls in category.tools
    }
    assert "broken" not in all_tool_names


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
        from arancio.core.plugins.base import Plugin
        from arancio.core.plugins.tool import tool


        class Doubler(Plugin):
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

    result, hook_messages = tools["double"].call(call_id="c1", x=21)

    assert hook_messages == []
    assert isinstance(result, ToolResultMessage)
    assert result.content == 42
