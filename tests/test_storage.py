"""Tests for the storage manager."""

from pathlib import Path

from arancio.storage.manager import StorageManager


def test_make_dir_creates_nested_directory_under_root(tmp_path: Path) -> None:
    """make_dir creates a nested directory (and parents) under the root."""
    storage = StorageManager(root=tmp_path)

    result = storage.make_dir("foo", "bar")

    assert result == tmp_path / "foo" / "bar"
    assert result.is_dir()


def test_bind_litellm_login_dir_symlinks_when_absent(tmp_path: Path) -> None:
    """An absent config dir is created as a symlink to ``<root>/litellm``."""
    root = tmp_path / "arancio"
    config_dir = tmp_path / "config" / "litellm"

    target = StorageManager(root=root).bind_litellm_login_dir(
        litellm_config_dir=config_dir
    )

    assert target == root / "litellm"
    assert target.is_dir()
    assert config_dir.is_symlink()
    assert config_dir.resolve() == target.resolve()


def test_bind_litellm_login_dir_migrates_existing_dir(tmp_path: Path) -> None:
    """An existing real config dir is migrated into the target, then symlinked."""
    root = tmp_path / "arancio"
    config_dir = tmp_path / "config" / "litellm"
    (config_dir / "chatgpt").mkdir(parents=True)
    (config_dir / "chatgpt" / "auth.json").write_text("{}")

    target = StorageManager(root=root).bind_litellm_login_dir(
        litellm_config_dir=config_dir
    )

    assert config_dir.is_symlink()
    assert (target / "chatgpt" / "auth.json").read_text() == "{}"
    # the migrated login stays reachable through the symlink
    assert (config_dir / "chatgpt" / "auth.json").read_text() == "{}"


def test_bind_litellm_login_dir_noop_when_already_symlink(tmp_path: Path) -> None:
    """An already-symlinked config dir is left untouched."""
    root = tmp_path / "arancio"
    storage = StorageManager(root=root)
    target = storage.make_dir("litellm")
    config_dir = tmp_path / "config" / "litellm"
    config_dir.parent.mkdir(parents=True)
    config_dir.symlink_to(target, target_is_directory=True)

    result = storage.bind_litellm_login_dir(litellm_config_dir=config_dir)

    assert result == target
    assert config_dir.is_symlink()
    assert config_dir.resolve() == target.resolve()
