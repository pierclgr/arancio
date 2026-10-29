"""Shared filesystem path constants."""

from pathlib import Path

# arancio working directory holding all persistent state
ARANCIO_DEFAULT_DIR = Path.home() / ".arancio"

# harness resources; the user places their files here manually
HARNESS_DIR_ROOT_PATH = ARANCIO_DEFAULT_DIR / "harness"
