"""Pytest session setup: point the harness path at the repo's checked-in copy.

Tools and the system prompt builder read their harness files from the arancio working
directory (``~/.arancio/harness``), which a real run expects the user to have populated.
Tests must not depend on the real home, so this module repoints the harness path
constants they read at the repository's checked-in ``harness/`` before any test module
is imported.
"""

from pathlib import Path

from arancio.core.builders import system_prompt as system_prompt_builder
from arancio.core.tools import base as tools_base

_REPO_HARNESS = Path(__file__).resolve().parent.parent / "harness"

tools_base.TOOLS_HARNESS_PATH = _REPO_HARNESS / "tools"
system_prompt_builder.SYSTEM_PROMPT_HARNESS_PATH = _REPO_HARNESS / "SYSTEM_PROMPT.md"
