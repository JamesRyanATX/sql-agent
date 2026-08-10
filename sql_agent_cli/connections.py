"""Registering databases, and choosing which one you are talking to.

The registry lives on the server — these are its client. `connect` is the one
piece of state the CLI keeps for itself: which of the registered connections
this user is currently asking about.
"""

from __future__ import annotations

import sys

import click

from sql_agent_cli import config, http, render


@click.command()
@click.argument("connection_id", metavar="ID", required=False)
def connect(connection_id: str | None) -> None:
    """Choose the database everything else talks to.

    Nothing is actually connected — this picks a default, the way `kubectl
    config use-context` does, and stores it under $XDG_STATE_HOME. With no ID it
    prints the current choice.
    """
    http.run(_connect(connection_id))


async def _connect(connection_id: str | None) -> None:
    if connection_id is None:
        click.echo(config.connection(None))
        return
    # Validated before it is written. One cheap request stops a typo surfacing
    # on the *next* command, where it reads like a server problem rather than a
    # misspelling — and it gives `connect` something worth printing back.
    body = await http.get(f"/connections/{connection_id}")
    config.select(connection_id)
    click.echo(f"connected to {click.style(body['id'], bold=True)} — {body['dsn']}")


@click.group()
def connections() -> None:
    """Manage the databases the agent can be pointed at."""


@connections.command("ls")
def list_connections() -> None:
    """Every registered database."""
    http.run(_list())


async def _list() -> None:
    body = await http.get("/connections")
    rows = body["connections"]
    if not rows:
        click.echo("no connections — register one with 'sql-agent connections create'")
        return
    current = config.selected()
    render.table(
        ["", "id", "driver", "host", "port", "database", "username", "cache", "turns"],
        [
            [
                "*" if r["id"] == current else "",
                r["id"], r["driver"], r["host"], r["port"], r["database"],
                r["username"], r["cache_entries"], r["turns"],
            ]
            for r in rows
        ],
        right=frozenset({"port", "cache", "turns"}),
    )


@connections.command("get")
@click.argument("connection_id", metavar="ID")
def get_connection(connection_id: str) -> None:
    """Everything about one, except the password."""
    http.run(_get(connection_id))


async def _get(connection_id: str) -> None:
    body = await http.get(f"/connections/{connection_id}")
    click.echo(click.style(body["id"], bold=True))
    render.detail(
        [
            ("label", body["label"]),
            ("driver", body["driver"]),
            ("host", body["host"]),
            ("port", body["port"]),
            ("database", body["database"]),
            ("username", body["username"]),
            ("sslmode", body["sslmode"]),
            ("read-only", body["readonly_tier"]),
            # Never the value, and never a masked length either — a length is a
            # fact about the password.
            ("password", "(set)" if body["has_password"] else "(unset)"),
            ("owner", "the environment" if body["origin"] == "env" else "you"),
            ("cache", f"{body['cache_entries']} entries"),
            ("turns", body["turns"]),
        ]
    )


def _password_options(f):
    for option in reversed(
        [
            click.option(
                "--password",
                help="Passing it here puts it in your shell history and in `ps`. "
                "Prefer --password-stdin.",
            ),
            click.option(
                "--password-stdin",
                is_flag=True,
                help="Read the password from the first line of stdin.",
            ),
            click.option(
                "--password-prompt",
                is_flag=True,
                help="Ask for it, hidden and confirmed.",
            ),
        ]
    ):
        f = option(f)
    return f


def resolve_password(value, from_stdin, prompt, *, required: bool) -> str | None:
    """None means "leave it alone" — only `update` may get that back.

    Three flags for one field looks like a lot until you try to fold them: the
    literal has to exist because it was asked for, stdin has to exist because
    scripts have no terminal, and the prompt has to exist because it is the only
    one of the three that does not write the password down somewhere. `psql`
    resolves this by refusing the literal outright, which is the better design
    and not the one specified.
    """
    given = [
        flag
        for flag, on in (
            ("--password", value is not None),
            ("--password-stdin", from_stdin),
            ("--password-prompt", prompt),
        )
        if on
    ]
    if len(given) > 1:
        raise click.UsageError(f"{' and '.join(given)} are answers to one question")
    if from_stdin:
        return sys.stdin.readline().rstrip("\n")
    if value is not None:
        return value
    if prompt or required:
        if not sys.stdin.isatty():
            raise click.UsageError(
                "no password, and stdin is not a terminal — use --password-stdin"
            )
        return click.prompt("password", hide_input=True, confirmation_prompt=True)
    return None


@connections.command("create")
@click.argument("connection_id", metavar="ID")
@click.option(
    "--driver",
    type=click.Choice(["postgresql+psycopg", "mysql+asyncmy", "sqlite+aiosqlite"]),
    default="postgresql+psycopg",
    show_default=True,
    help="Which engine. SQLite needs only --database, the file path.",
)
@click.option("--hostname", help="Host the database is on. Not for sqlite.")
@click.option("--database", required=True, help="Database name, or a file path for sqlite.")
@click.option("--username", help="Role to connect as. Not for sqlite.")
@click.option("--port", type=int, help="Defaults to the driver's usual port.")
@click.option("--label", help="A human note. Shown by `connections get`.")
@click.option("--sslmode", default="prefer", show_default=True)
@_password_options
@click.option("--no-test", is_flag=True, help="Skip the connection check.")
def create_connection(
    connection_id, driver, hostname, database, username, port, label, sslmode,
    password, password_stdin, password_prompt, no_test,
) -> None:
    """Register a database the agent can be pointed at.

    ID is the name used everywhere else — 'warehouse', 'prod'.

    The agent needs SELECT on this database and nothing more; give it a
    read-only role. With no password flag you are prompted, so it stays out of
    your shell history and out of everyone else's `ps`.
    """
    # SQLite has no host, user or password, and the server rejects them rather
    # than ignoring them — so do not prompt for a password it cannot use.
    sqlite = driver.startswith("sqlite")
    secret = (
        None if sqlite
        else resolve_password(password, password_stdin, password_prompt, required=True)
    )
    body: dict = {"id": connection_id, "driver": driver, "database": database}
    if not sqlite:
        body |= {"host": hostname, "username": username, "sslmode": sslmode,
                 "password": secret}
        if port is not None:
            body["port"] = port
    if label:
        body["label"] = label
    http.run(_create(body, probe=not no_test))


async def _create(body: dict, *, probe: bool) -> None:
    result = await http.post(
        "/connections", json=body, params={"probe": str(probe).lower()}
    )
    cid = result["connection"]["id"]
    click.echo(f"registered {click.style(cid, bold=True)} — {result['connection']['dsn']}")
    _show_test(result.get("test"))

    # Removes a step from the quickstart, and cannot surprise anyone who already
    # made a choice.
    if config.selected() is None:
        config.select(cid)
        click.echo(f"connected to {click.style(cid, bold=True)}")


def _show_test(test: dict | None) -> None:
    if test is None:
        return
    if not test["ok"]:
        click.secho(f"  cannot reach it: {test['error']}", fg="red")
        click.secho("  registered anyway — fix it with 'connections update'", dim=True)
        return
    who = f"as {test['username']} " if test.get("username") else ""
    click.secho(
        f"  reached it in {test['latency_ms']}ms {who}— {test['tables']} tables "
        f"in {test.get('default_schema') or 'the default schema'}",
        dim=True,
    )
    if test.get("readonly_tier") == "partial":
        click.secho("  read-only: partial on this engine", fg="yellow")
    for warning in test["warnings"]:
        click.secho(f"  warning: {warning}", fg="yellow")


@connections.command("update")
@click.argument("connection_id", metavar="ID")
@click.option("--hostname")
@click.option("--database")
@click.option("--username")
@click.option("--port", type=int)
@click.option("--label")
@click.option("--sslmode")
@_password_options
def update_connection(
    connection_id, hostname, database, username, port, label, sslmode,
    password, password_stdin, password_prompt,
) -> None:
    """Change one or more fields of a connection.

    Only the flags you pass are sent — everything else is left alone, and an
    omitted --password keeps the stored one rather than clearing it.
    """
    secret = resolve_password(password, password_stdin, password_prompt, required=False)
    body = {
        k: v
        for k, v in (
            ("host", hostname), ("database", database), ("username", username),
            ("port", port), ("label", label), ("sslmode", sslmode),
            ("password", secret),
        )
        if v is not None
    }
    if not body:
        raise click.UsageError(
            "nothing to update — pass one of --hostname --database --username "
            "--port --label --sslmode --password"
        )
    http.run(_update(connection_id, body))


async def _update(connection_id: str, body: dict) -> None:
    result = await http.patch(f"/connections/{connection_id}", json=body)
    click.echo(f"updated {click.style(result['id'], bold=True)} — {result['dsn']}")


@connections.command("rm")
@click.argument("connection_id", metavar="ID")
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation.")
def remove_connection(connection_id: str, yes: bool) -> None:
    """Forget a database, and everything the agent learned about it."""
    if not yes:
        # It takes the learned cache with it, and the id it takes is one
        # mistyped argument away from a different database.
        click.confirm(
            f"remove {connection_id!r} and everything learned about it?", abort=True
        )
    http.run(_remove(connection_id))


async def _remove(connection_id: str) -> None:
    wiped = (await http.delete(f"/connections/{connection_id}"))["wiped"]
    click.echo(f"removed {click.style(connection_id, bold=True)}")
    for table, rows in sorted(wiped.items()):
        if rows:
            click.secho(f"  wiped {table} ({rows} rows)", dim=True)
    # Otherwise the state file points at a 404 and every later command fails
    # with a message about a connection the user just deleted.
    if config.selected() == connection_id:
        config.select(None)
        click.secho("  no connection selected now", dim=True)
