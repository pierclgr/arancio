"""Pytest setup: keep every test off the user's real arancio directory.

Three things in this project reach outside a test by default. Tools and the system
prompt builder read their harness files from ``~/.arancio/harness``; settings and the
LiteLLM login directory are written through module constants that ignore a
:class:`StorageManager`'s ``root``; and the read-first tool guard is a process-wide
singleton that outlives any one test. This module repoints the first, isolates the
second and resets the third.

The harness repointing happens at import, because pytest imports this module before any
test module, and importing :mod:`arancio.prompt.actions.executor` instantiates two tools
as class attributes.
"""

from pathlib import Path
from typing import Iterator

import pytest
from fakes import ScriptedClient, ScriptedController

from arancio.core.agents import Agent
from arancio.core.builders import system_prompt as system_prompt_builder
from arancio.core.permissions.manager import PermissionManager
from arancio.core.tools import base as tools_base
from arancio.core.tools.manager import ToolManager
from arancio.core.tools.session import shared_session
from arancio.sessions.manager import SessionManager
from arancio.settings.manager import SettingsManager
from arancio.storage import manager as storage_module
from arancio.storage.manager import StorageManager

_REPO_HARNESS = Path(__file__).resolve().parent.parent / "harness"

tools_base.TOOLS_HARNESS_PATH = _REPO_HARNESS / "tools"
system_prompt_builder.SYSTEM_PROMPT_HARNESS_PATH = _REPO_HARNESS / "SYSTEM_PROMPT.md"


@pytest.fixture(autouse=True)
def _isolate_arancio_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the two absolute paths a storage manager's root does not cover.

    ``save_settings``/``load_settings`` and ``bind_litellm_login_dir`` read these module
    globals at call time, so without this a test writes the real
    ``~/.arancio/settings.yml`` or moves the real ChatGPT login directory.
    """
    monkeypatch.setattr(
        storage_module, "ARANCIO_SETTINGS_FILE", tmp_path / "settings.yml"
    )
    monkeypatch.setattr(storage_module, "ARANCIO_LITELLM_DIR", tmp_path / "litellm")


@pytest.fixture(autouse=True)
def _reset_tool_session() -> Iterator[None]:
    """Empty the shared read guard before and after every test.

    Yields:
        Control to the test, with the singleton known to be empty.
    """
    shared_session.clear()
    yield
    shared_session.clear()


@pytest.fixture
def controller() -> ScriptedController:
    """Return a controller that allows every permission request.

    Returns:
        A controller recording each request it is sent.
    """
    return ScriptedController()


@pytest.fixture
def client() -> ScriptedClient:
    """Return a client that answers with a plain assistant message.

    Returns:
        A client recording each request it is sent.
    """
    return ScriptedClient()


@pytest.fixture
def storage_manager(tmp_path: Path) -> StorageManager:
    """Return a storage manager rooted inside the test's temporary directory.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A storage manager rooted at ``<tmp_path>/arancio``.
    """
    return StorageManager(root=tmp_path / "arancio")


@pytest.fixture
def tool_manager(client: ScriptedClient) -> ToolManager:
    """Return a tool manager whose web summary client is scripted.

    Args:
        client: the scripted client handed to ``FetchWebTool``.

    Returns:
        A tool manager building real tools from the repo harness.
    """
    return ToolManager(web_summary_client=client)


@pytest.fixture
def permission_manager(
    tool_manager: ToolManager, controller: ScriptedController
) -> PermissionManager:
    """Return a permission manager with every category at ``ASK``.

    Args:
        tool_manager: the manager that builds the granted tools.
        controller: the controller asked to resolve each ``ASK`` call.

    Returns:
        A permission manager over real tools.
    """
    return PermissionManager(tool_manager=tool_manager, controller=controller)


@pytest.fixture
def agent(client: ScriptedClient, permission_manager: PermissionManager) -> Agent:
    """Return a real agent driven by the scripted client.

    Args:
        client: the scripted client standing in for the model.
        permission_manager: the manager gating the agent's tool calls.

    Returns:
        An agent with an empty history.
    """
    return Agent(client=client, permission_manager=permission_manager)


@pytest.fixture
def session_manager(storage_manager: StorageManager) -> SessionManager:
    """Return a session manager writing under the temporary storage root.

    It shares the process-wide tool session, as production does, which the
    autouse reset keeps clean between tests.

    Args:
        storage_manager: the storage manager doing the file I/O.

    Returns:
        A session manager with no session open yet.
    """
    return SessionManager(storage_manager, root=storage_manager.root)


@pytest.fixture
def settings_manager(
    storage_manager: StorageManager, client: ScriptedClient, agent: Agent
) -> SettingsManager:
    """Return a settings manager wired to the scripted client and real agent.

    Args:
        storage_manager: the storage manager persisting settings.
        client: the scripted client used as both main and summary client.
        agent: the agent whose loop limits settings are applied to.

    Returns:
        A settings manager holding default settings, not yet loaded.
    """
    return SettingsManager(
        storage_manager=storage_manager,
        client=client,
        summary_client=client,
        agent=agent,
    )
