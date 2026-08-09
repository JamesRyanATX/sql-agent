"""The CLI is a client of the API, not a second way into it.

`app/api.py`'s docstring says the graph, the pool and the checkpointer exist in
one process and that the code path the demo exercises is the code path a user
gets. That only stays true while the CLI cannot reach around the HTTP boundary,
and "don't import app" is the kind of rule that holds until somebody needs one
convenient function at 6pm.

So it is a test, in the register of tests/test_traps.py: the packaging already
makes it true (the wheel ships `sql_agent_cli` and nothing else), and this says
so out loud in the place where it would be broken.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

CLI = pathlib.Path(__file__).resolve().parent.parent / "sql_agent_cli"
MODULES = sorted(CLI.glob("*.py"))


def imported_names(path: pathlib.Path) -> set[str]:
    """Every top-level package this module imports, however it imports it."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_there_is_a_cli_to_check():
    assert MODULES, "no modules found — did the package move?"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_the_cli_never_imports_the_server(path):
    assert "app" not in imported_names(path), (
        f"{path.name} imports `app` — the CLI talks to the server over HTTP. "
        "If you need something from it, it belongs behind an endpoint."
    )


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_the_cli_never_imports_a_database_driver(path):
    """The subtler version of the same rule. `scripts/` held no Python that
    touched a database at all, and neither does this."""
    assert not {"psycopg", "psycopg_pool", "sqlalchemy"} & imported_names(path), path.name
