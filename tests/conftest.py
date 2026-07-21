"""Pytest session setup: point the harness path at the repo's checked-in copy.

Tools read their harness files from the arancio working directory
(``~/.arancio/harness``), which a real run expects the user to have populated. Tests
must not depend on the real home, so this module repoints the harness path constants the
tools read at the repository's checked-in ``harness/`` before any test module is
imported.
"""

from pathlib import Path

from arancio.core.tools import base as tools_base

_REPO_HARNESS = Path(__file__).resolve().parent.parent / "harness"

tools_base.TOOLS_HARNESS_PATH = _REPO_HARNESS / "tools"
