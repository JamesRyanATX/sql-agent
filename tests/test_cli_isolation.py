"""Who is allowed to import what.

The original rule: the CLI is a client of the API, not a second way into it.
`app/api.py`'s docstring says the graph, the pool and the checkpointer exist in
one process and that the code path the demo exercises is the code path a user
gets. That only stays true while the CLI cannot reach around the HTTP boundary,
and "don't import app" is the kind of rule that holds until somebody needs one
convenient function at 6pm.

So it is a test, in the register of tests/test_traps.py: the packaging already
makes it true (the wheel ships `sql_agent` and nothing else), and this says
so out loud in the place where it would be broken.

Two more live here now, on the same argument. `langfuse` is import-legal in
`app/tracing.py` alone — that was documentation only until `tracing` grew a read
half and gave a second module a reason to want its own client. And nothing that
ships imports `optim/`, which is a development tool in the same category as
pytest: it reads the app, the app never reads it.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLI = ROOT / "cli" / "sql_agent"
APP = ROOT / "app"
MODULES = sorted(CLI.glob("*.py"))
SHIPPED = sorted(APP.glob("*.py")) + MODULES


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


@pytest.mark.parametrize("path", sorted(APP.glob("*.py")), ids=lambda p: p.name)
def test_only_tracing_imports_langfuse(path):
    """CLAUDE.md has said this since tracing landed; nothing enforced it.

    The rule's purpose is that there is one `enabled()` to read rather than a
    scattering of `if` at every call site — so a second module constructing its
    own client is exactly the failure, and the temptation is now real: the
    optimiser needs to *read* traces back, and `app/tracing.py` is where that
    read half goes rather than wherever it is first wanted.
    """
    assert path.name == "tracing.py" or "langfuse" not in imported_names(path), (
        f"{path.name} imports langfuse — it belongs behind a helper in "
        "app/tracing.py, which is the one module that knows Langfuse exists."
    )


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_nothing_that_ships_imports_the_optimiser(path):
    """`optim/` reads the app; the app never reads `optim/`.

    It is a development tool, `gepa` is a dependency group the image does not
    install, and an `app/` module importing it would turn a missing dev
    dependency into a server that will not boot.
    """
    assert "optim" not in imported_names(path), (
        f"{path.name} imports `optim` — the arrow points the other way. "
        "The optimiser is a dev tool, like pytest."
    )
