"""Tests asserting ``arancio.core`` stands on its own.

``arancio.core`` is a library the arancio wrapper is built on: the wrapper may depend on
core, never the reverse. These tests fail the moment a core module reaches back into
``arancio.sessions``, ``arancio.ui`` or any other wrapper package.
"""

import ast
import subprocess
import sys
from pathlib import Path

CORE_PACKAGE = "arancio.core"
CORE_ROOT = Path(__file__).resolve().parents[1] / "src" / "arancio" / "core"


def _imported_modules(path: Path) -> list[str]:
    """Return every module one source file imports, resolving relative imports.

    Args:
        path: the source file to scan.

    Returns:
        The absolute dotted module names the file imports, in source order.
    """
    # the file's own package, used to resolve a relative import to an absolute name
    package = ".".join(
        (CORE_PACKAGE, *path.relative_to(CORE_ROOT).parent.parts)
    ).rstrip(".")
    modules: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if not node.level:
                modules.append(node.module or "")
                continue
            # a level of 1 means the file's own package, each extra one climbs
            base = package.split(".")[: len(package.split(".")) - node.level + 1]
            modules.append(".".join((*base, node.module or "")).rstrip("."))
    return modules


def test_no_core_module_imports_the_wrapper() -> None:
    """No file under ``arancio/core`` imports an ``arancio`` package outside core."""
    offenders = [
        f"{path.relative_to(CORE_ROOT)}: {module}"
        for path in sorted(CORE_ROOT.rglob("*.py"))
        for module in _imported_modules(path)
        if module.split(".")[0] == "arancio"
        and module != CORE_PACKAGE
        and not module.startswith(f"{CORE_PACKAGE}.")
    ]

    assert offenders == []


def test_importing_core_loads_no_wrapper_module() -> None:
    """A fresh interpreter importing core pulls in no wrapper module at runtime.

    Catches what the source scan cannot: a deferred import inside a function
    body, or a module pulled in dynamically.
    """
    script = (
        "import sys, arancio.core.agents, arancio.core.permissions.manager;"
        "print(sorted(m for m in sys.modules"
        " if m.startswith('arancio.')"
        " and not m.startswith('arancio.core')))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"
