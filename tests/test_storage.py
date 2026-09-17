"""Tests for the file-I/O layer the session log and settings are built on.

``StorageManager`` is deliberately dumb: it does no validation and holds no policy. What
it does own is the append-and-recover primitives the session recorder relies on, and two
methods that write outside the manager's ``root`` through module constants — which is
why ``conftest`` redirects those.
"""

from pathlib import Path

import pytest
import yaml

import arancio.storage.manager as storage_module
from arancio.core.messages import ErrorMessage, WarningMessage
from arancio.settings.settings import Settings
from arancio.storage.manager import StorageManager


def test_a_relative_directory_is_created_under_the_root(tmp_path: Path) -> None:
    """A relative path is the only case the root applies to."""
    storage = StorageManager(root=tmp_path / "arancio")

    created = storage.make_dir("sessions/2026")

    assert created == tmp_path / "arancio" / "sessions" / "2026"
    assert created.is_dir()


def test_an_absolute_directory_bypasses_the_root(tmp_path: Path) -> None:
    """An absolute path is used as given, which is how the constants escape."""
    storage = StorageManager(root=tmp_path / "arancio")

    created = storage.make_dir(tmp_path / "elsewhere")

    assert created == tmp_path / "elsewhere"


def test_creating_a_file_twice_is_refused(tmp_path: Path) -> None:
    """Exclusive creation is what stops a session log overwriting another.

    The recorder uses it for the first line of a new log, so a collision has to fail
    rather than silently truncate an existing chat.
    """
    path = tmp_path / "a.jsonl"
    StorageManager.create_file(path, "first")

    with pytest.raises(FileExistsError):
        StorageManager.create_file(path, "second")

    assert path.read_text() == "first\n"


def test_lines_are_appended_with_their_terminator(tmp_path: Path) -> None:
    """Each record is one line, so the newline is the storage layer's job."""
    path = tmp_path / "a.jsonl"
    StorageManager.create_file(path, "one")
    StorageManager.append_line(path, "two")

    assert StorageManager.read_lines(path) == ["one", "two"]


def test_a_missing_file_has_no_size(tmp_path: Path) -> None:
    """The recovery offset of the first write has to be zero, not an error."""
    assert StorageManager.file_size(tmp_path / "absent.jsonl") == 0


def test_truncating_drops_everything_past_the_offset(tmp_path: Path) -> None:
    """This is how a half-written record is rolled back before a retry."""
    path = tmp_path / "a.jsonl"
    StorageManager.create_file(path, "keep")
    offset = StorageManager.file_size(path)
    StorageManager.append_line(path, "drop")

    StorageManager.truncate_file(path, offset)

    assert StorageManager.read_lines(path) == ["keep"]


def test_removing_an_absent_file_is_not_an_error(tmp_path: Path) -> None:
    """Cleanup after a failed first write must not fail in turn."""
    StorageManager.remove_file(tmp_path / "absent.jsonl")


def test_a_missing_settings_file_is_created_and_reported(tmp_path: Path) -> None:
    """A first run leaves a populated file behind but hands back no content.

    Returning ``None`` rather than the defaults is deliberate: deciding what "missing"
    falls back to belongs to the settings manager.
    """
    data, messages = StorageManager.load_settings()

    assert data is None
    assert isinstance(messages[0], WarningMessage)
    assert storage_module.ARANCIO_SETTINGS_FILE.exists()


@pytest.mark.parametrize(
    "contents",
    ["", "- a\n- b\n", "key: [unclosed\n"],
    ids=["empty", "top-level-list", "syntax-error"],
)
def test_an_unusable_settings_file_is_reported_and_left_alone(contents: str) -> None:
    """A broken file is never rewritten, so its error resurfaces until fixed."""
    storage_module.ARANCIO_SETTINGS_FILE.write_text(contents)

    data, messages = StorageManager.load_settings()

    assert data is None
    assert isinstance(messages[0], ErrorMessage)
    assert storage_module.ARANCIO_SETTINGS_FILE.read_text() == contents


def test_a_well_formed_settings_file_is_returned_unvalidated() -> None:
    """Field-level checking happens later; this layer only parses."""
    storage_module.ARANCIO_SETTINGS_FILE.write_text("max_retries: not-a-number\n")

    data, messages = StorageManager.load_settings()

    assert data == {"max_retries": "not-a-number"}
    assert messages == []


def test_saved_settings_round_trip_through_yaml() -> None:
    """What the manager writes is what a later load parses back."""
    StorageManager.save_settings(Settings.default())

    written = yaml.safe_load(storage_module.ARANCIO_SETTINGS_FILE.read_text())

    assert written["max_turns"] == "inf"
    assert written["permissions"] == {
        "read": "ask",
        "write": "ask",
        "web": "ask",
        "execute": "ask",
    }


def test_an_absent_litellm_config_dir_becomes_a_symlink(tmp_path: Path) -> None:
    """LiteLLM hardcodes its config path, so it is redirected by symlink."""
    storage = StorageManager(root=tmp_path / "arancio")
    config_dir = tmp_path / "config" / "litellm"

    target = storage.bind_litellm_login_dir(litellm_config_dir=config_dir)

    assert config_dir.is_symlink()
    assert config_dir.resolve() == target.resolve()


def test_an_existing_login_directory_is_migrated_not_discarded(
    tmp_path: Path,
) -> None:
    """A real sign-in already on disk has to survive the redirect."""
    storage = StorageManager(root=tmp_path / "arancio")
    config_dir = tmp_path / "config" / "litellm"
    (config_dir / "chatgpt").mkdir(parents=True)
    (config_dir / "chatgpt" / "auth.json").write_text("{token}")

    target = storage.bind_litellm_login_dir(litellm_config_dir=config_dir)

    assert (target / "chatgpt" / "auth.json").read_text() == "{token}"
    assert (config_dir / "chatgpt" / "auth.json").read_text() == "{token}"
    assert config_dir.is_symlink()


def test_an_already_bound_config_dir_is_left_alone(tmp_path: Path) -> None:
    """Re-binding on every launch must be a no-op, not a repeated migration."""
    storage = StorageManager(root=tmp_path / "arancio")
    config_dir = tmp_path / "config" / "litellm"
    first = storage.bind_litellm_login_dir(litellm_config_dir=config_dir)
    (first / "marker").write_text("x")

    second = storage.bind_litellm_login_dir(litellm_config_dir=config_dir)

    assert second == first
    assert (second / "marker").read_text() == "x"
