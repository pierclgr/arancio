"""Tests for the SettingsManager facade: apply, load and save."""

import arancio.storage.manager as storage_mod
from arancio.core.agents import Agent
from arancio.core.clients.litellm import LiteLLMClient
from arancio.core.permissions.manager import PermissionManager
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.core.tools.manager import ToolManager
from arancio.settings.manager import SettingsManager
from arancio.settings.settings import Settings
from arancio.storage.manager import StorageManager


class _FakeController:
    """Controller stub that fails if asked (apply never gates a call)."""

    def request(self, request):
        """Fail the test: settings application must not prompt the user.

        Args:
            request: the (unexpected) controller request.

        Raises:
            AssertionError: always, since no request is expected.
        """
        raise AssertionError("controller.request must not be called")


def _build():
    """Build a SettingsManager over a real client, summary client and agent.

    Returns:
        The manager and the live objects it configures.
    """
    client = LiteLLMClient(stream=True)
    summary_client = LiteLLMClient(stream=False)
    # mirror the entry points: the summary client is built with thinking disabled
    summary_client.thinking_effort = None
    summary_client.thinking_summary = None
    tool_manager = ToolManager(web_summary_client=summary_client)
    permission_manager = PermissionManager(tool_manager, _FakeController())
    agent = Agent(client=client, permission_manager=permission_manager)
    manager = SettingsManager(
        storage=StorageManager(),
        client=client,
        summary_client=summary_client,
        agent=agent,
    )
    return manager, client, summary_client, agent


def _custom_settings():
    """Return a non-default settings snapshot used across tests."""
    return Settings(
        permissions={PermissionCategory.READ: PermissionLevel.AUTO},
        provider="openai",
        model_name="gpt-4o",
        thinking_effort="high",
        thinking_summary=None,
        max_turns=9,
        max_retries=3,
        turn_wait_time=5.0,
        turn_wait_time_multiplier=4.0,
    )


def test_apply_pushes_settings_into_live_objects():
    """``apply`` writes every field onto the client, summary client and agent."""
    manager, client, summary_client, agent = _build()
    manager.settings = _custom_settings()

    manager.apply()

    assert client.model_id == "openai/gpt-4o"
    assert client.thinking_effort == "high"
    assert client.thinking_summary is None
    assert summary_client.model_id == "openai/gpt-4o"
    # apply leaves the summary client's (construction-time) thinking untouched
    assert summary_client.thinking_effort is None
    assert summary_client.thinking_summary is None
    assert agent.max_turns == 9
    assert agent.max_retries == 3
    assert agent.retry_delay == 5.0
    assert agent.retry_delay_multiplier == 4.0
    # only READ granted: the catalog rebuilds to the read tools alone
    tool_names = {type(tool).__name__ for tool in agent._tools.values()}
    assert tool_names == {"ReadFileTool", "GlobTool", "GrepTool"}


def test_load_creates_defaults_and_applies(monkeypatch, tmp_path):
    """``load`` builds the default file when absent and applies the defaults."""
    monkeypatch.setattr(storage_mod, "ARANCIO_SETTINGS_FILE", tmp_path / "settings.yml")
    manager, client, _, agent = _build()

    loaded = manager.load()

    assert (tmp_path / "settings.yml").exists()
    assert loaded == Settings.default()
    assert client.model_id is None
    assert agent.max_turns == Settings.default().max_turns
    # every category granted at ASK: the full tool catalog is built
    assert len(agent._tools) == 9


def test_save_persists_current_settings(monkeypatch, tmp_path):
    """``save`` writes the current settings so a later load reads them back."""
    monkeypatch.setattr(storage_mod, "ARANCIO_SETTINGS_FILE", tmp_path / "settings.yml")
    manager, *_ = _build()
    manager.settings = _custom_settings()

    manager.save()

    assert StorageManager().load_settings() == _custom_settings()
