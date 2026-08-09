"""Where the CLI is pointed, and which connection it is pointed at.

Two environment variables and one small state file is the whole of it. Nothing
under `sql_agent_cli/` imports `app` — the CLI is an HTTP client of the server,
not a second way into it — so these names are the CLI's own rather than the
server's, and the 401 message names both sides.
"""

from __future__ import annotations

import json
import os
import pathlib

import click

from sql_agent_cli.http import ApiError

ENV_URL = "SQL_AGENT_URL"
ENV_KEY = "SQL_AGENT_API_KEY"
ENV_CONNECTION = "SQL_AGENT_CONNECTION"

# What `make up` starts. Defaulted rather than required for the same reason
# every URL in app/settings.py is: a fresh clone should work after `make up`
# without a setup step, and "no API at http://localhost:8000/v1" is a better
# first error than "you have not configured me".
DEFAULT_URL = "http://localhost:8000/v1"


def base_url() -> str:
    """The server, including its `/v1` prefix.

    httpx keeps a base_url's path when it merges a relative one, so a call site
    writes `/cache` and gets `/v1/cache`. Stripping the prefix here instead
    would 404 every request against a server running perfectly.
    """
    return os.environ.get(ENV_URL, "").strip() or DEFAULT_URL


def api_key() -> str:
    return os.environ.get(ENV_KEY, "")


def state_path() -> pathlib.Path:
    root = os.environ.get("XDG_STATE_HOME") or pathlib.Path.home() / ".local" / "state"
    return pathlib.Path(root) / "sql-agent" / "state.json"


def selected() -> str | None:
    """The connection `sql-agent connect` last chose, if any."""
    try:
        return json.loads(state_path().read_text()).get("connection")
    except (OSError, ValueError):
        # A missing file is the normal first run. A corrupt one is not worth a
        # traceback either — the next `sql-agent connect` overwrites it, and the
        # message a user gets is the same one they get with no file at all.
        return None


def select(cid: str | None) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"connection": cid} if cid else {}) + "\n")


def connection(flag: str | None) -> str:
    """-c, then the environment, then `sql-agent connect`, then an error.

    The environment sits in the middle so a CI job or a `direnv` can retarget
    every command in one shell without writing to a state file another process
    may be reading.

    State rather than config, and `~/.local/state` rather than `~/.config`:
    nobody authors this file, it changes as a side effect of a command, and
    losing it costs exactly one `sql-agent connect`.
    """
    cid = flag or os.environ.get(ENV_CONNECTION) or selected()
    if not cid:
        raise ApiError(
            "no connection selected — pick one with 'sql-agent connect <id>' "
            "(see 'sql-agent connections ls')"
        )
    return cid


def option(f):
    """The `-c/--connection` flag, on every command that needs one.

    Lives here rather than in `main` so the command modules can import it
    without importing the group that imports them.
    """
    return click.option(
        "-c",
        "--connection",
        "connection",
        metavar="ID",
        help="Which registered database. Defaults to `sql-agent connect`'s.",
    )(f)
