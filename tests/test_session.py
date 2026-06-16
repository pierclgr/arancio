"""Tests for the shared tool session registry."""

from codo.core.tools.session import ToolSession


def test_session_records_and_recalls_reads() -> None:
    """Recorded reads are visible via ``is_known`` and ``is_fresh``."""
    session = ToolSession()
    session.record_read(path="/abs/file.txt", mtime=12345.0)

    assert session.is_known("/abs/file.txt") is True
    assert session.is_fresh("/abs/file.txt", current_mtime=12345.0) is True


def test_session_reports_unknown_paths() -> None:
    """Unknown paths fail both presence and freshness checks."""
    session = ToolSession()

    assert session.is_known("/abs/missing.txt") is False
    assert session.is_fresh("/abs/missing.txt", current_mtime=0.0) is False


def test_session_detects_mtime_drift() -> None:
    """A drifted mtime invalidates the freshness check while presence stays True."""
    session = ToolSession()
    session.record_read(path="/abs/file.txt", mtime=12345.0)

    assert session.is_known("/abs/file.txt") is True
    assert session.is_fresh("/abs/file.txt", current_mtime=99999.0) is False


def test_session_clear_drops_all_entries() -> None:
    """``clear`` resets the session to its empty initial state."""
    session = ToolSession()
    session.record_read(path="/abs/a.txt", mtime=1.0)
    session.record_read(path="/abs/b.txt", mtime=2.0)

    session.clear()

    assert session.is_known("/abs/a.txt") is False
    assert session.is_known("/abs/b.txt") is False
