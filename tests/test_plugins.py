"""Tests for plugin discovery, loading, hook attachment and failure isolation."""

import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from fakes import RecordingApp, ScriptedClient

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
from arancio.core.plugins.manager import PluginManager
from arancio.core.plugins.manifest import PluginManifestValidator
from arancio.prompt.actions.executor import ActionExecutor
from arancio.prompt.actions.types import PromptAction, ShellCommandAction
from arancio.sessions.manager import SessionManager
from arancio.settings.manager import SettingsManager
from arancio.storage.manager import StorageManager

_LOGGER_MODULE = """
from typing import ClassVar

from arancio.core.hooks.types import Hook
from arancio.core.plugins.base import Plugin

calls = []


class Logger(Plugin):
    hooks: ClassVar[frozenset[Hook]] = frozenset({Hook.AGENT_START})

    def execute(self, hook, **kwargs) -> None:
        calls.append((hook, kwargs))
"""

_SECOND_CLASS = """

class Second(Plugin):
    hooks: ClassVar[frozenset[Hook]] = frozenset({Hook.AGENT_END})

    def execute(self, hook, **kwargs) -> None:
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


def test_a_hook_plugin_with_no_hooks_is_an_error(
    plugins_root: Path, plugin_manager: PluginManager
) -> None:
    """A hook plugin binding nothing could never run, so it is refused."""
    _write_plugin(
        plugins_root,
        "no_hooks",
        module="""
        from arancio.core.plugins.base import Plugin


        class Idle(Plugin):
            def execute(self, hook, **kwargs) -> None:
                pass
        """,
    )

    messages = plugin_manager.load()

    assert plugin_manager.plugins == []
    assert len(messages) == 1
    assert isinstance(messages[0], ErrorMessage)
    assert "declares no hooks" in messages[0].content


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
        from typing import ClassVar

        from arancio.core.hooks.types import Hook
        from arancio.core.plugins.base import Plugin

        from .helpers import GREETING


        class Greeter(Plugin):
            hooks: ClassVar[frozenset[Hook]] = frozenset({Hook.AGENT_START})

            def execute(self, hook, **kwargs) -> None:
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
    """The plugin receives the hook that fired plus that hook's own kwargs."""
    directory = _write_plugin(plugins_root, "records_calls")

    assert plugin_manager.load() == []
    assert [plugin.name for plugin in plugin_manager.plugins] == ["records_calls"]
    assert plugin_manager.plugins[0].directory == directory

    message = UserMessage(content="hi")
    hook_manager.run(Hook.AGENT_START, message=message)

    assert _loaded_module("records_calls").calls == [
        (Hook.AGENT_START, {"message": message})
    ]


def test_a_plugin_bound_to_two_hooks_tells_them_apart(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """``HookManager.run`` never names the hook, so the manager passes it in."""
    _write_plugin(
        plugins_root,
        "two_hooks",
        module="""
        from typing import ClassVar

        from arancio.core.hooks.types import Hook
        from arancio.core.plugins.base import Plugin

        seen = []


        class Both(Plugin):
            hooks: ClassVar[frozenset[Hook]] = frozenset(
                {Hook.TURN_START, Hook.TURN_END}
            )

            def execute(self, hook, **kwargs) -> None:
                seen.append(hook)
        """,
    )

    assert plugin_manager.load() == []

    hook_manager.run(Hook.TURN_START, turn=0)
    hook_manager.run(Hook.TURN_END, turn=0)

    assert _loaded_module("two_hooks").seen == [Hook.TURN_START, Hook.TURN_END]


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
        from typing import ClassVar

        from arancio.core.hooks.types import Hook
        from arancio.core.plugins.base import Plugin


        class Broken(Plugin):
            hooks: ClassVar[frozenset[Hook]] = frozenset({Hook.AGENT_START})

            def execute(self, hook, **kwargs) -> None:
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
    """A raising plugin never runs again, and the user is told exactly once."""
    _write_plugin(
        plugins_root,
        "always_raises",
        module="""
        from typing import ClassVar

        from arancio.core.hooks.types import Hook
        from arancio.core.plugins.base import Plugin

        runs = []


        class Broken(Plugin):
            hooks: ClassVar[frozenset[Hook]] = frozenset({Hook.AGENT_START})

            def execute(self, hook, **kwargs) -> None:
                runs.append(hook)
                raise RuntimeError("boom")
        """,
    )

    assert plugin_manager.load() == []

    notices = hook_manager.run(Hook.AGENT_START, message=UserMessage(content="one"))
    assert hook_manager.run(Hook.AGENT_START, message=UserMessage(content="two")) == []

    assert _loaded_module("always_raises").runs == [Hook.AGENT_START]

    assert len(notices) == 1
    assert isinstance(notices[0], ErrorMessage)
    assert not notices[0].in_history
    assert "always_raises" in notices[0].content
    assert "boom" in notices[0].content


def test_a_failing_plugin_stops_running_on_its_other_hooks(
    plugins_root: Path, plugin_manager: PluginManager, hook_manager: HookManager
) -> None:
    """Disabling is per plugin, not per hook."""
    _write_plugin(
        plugins_root,
        "fails_then_quiet",
        module="""
        from typing import ClassVar

        from arancio.core.hooks.types import Hook
        from arancio.core.plugins.base import Plugin

        runs = []


        class Broken(Plugin):
            hooks: ClassVar[frozenset[Hook]] = frozenset(
                {Hook.TURN_START, Hook.TURN_END}
            )

            def execute(self, hook, **kwargs) -> None:
                runs.append(hook)
                raise RuntimeError("boom")
        """,
    )

    assert plugin_manager.load() == []

    hook_manager.run(Hook.TURN_START, turn=0)
    hook_manager.run(Hook.TURN_END, turn=0)

    assert _loaded_module("fails_then_quiet").runs == [Hook.TURN_START]


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
        from typing import ClassVar

        from arancio.core.hooks.types import Hook
        from arancio.core.plugins.base import Plugin


        class Broken(Plugin):
            hooks: ClassVar[frozenset[Hook]] = frozenset({Hook.TURN_START})

            def execute(self, hook, **kwargs) -> None:
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
class Broken(Plugin):
    hooks = frozenset({{Hook.{hook.name}}})
    def execute(self, **kwargs):
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
runs = []
class Broken(Plugin):
    hooks = frozenset({Hook.TURN_END})
    def execute(self, **kwargs):
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
class Broken(Plugin):
    hooks = frozenset({{Hook.{hook}}})
    def execute(self, **kwargs):
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
class Broken(Plugin):
    hooks = frozenset({{Hook.{hook}}})
    def execute(self, **kwargs):
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
