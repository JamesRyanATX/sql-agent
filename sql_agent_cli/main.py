"""`sql-agent` — ask a database questions, and watch it get cheaper.

A pure HTTP client of the server in `app/`. It holds no business logic and
imports nothing from `app`: the graph, the pools and the checkpointer exist in
one process, and the code path this exercises is the code path a user gets.
(tests/test_cli_isolation.py asserts the import rule.)

Configured by two environment variables:

    SQL_AGENT_URL=http://localhost:8000/v1    the server, /v1 included
    SQL_AGENT_API_KEY=...                     matches the server's API_TOKEN
"""

from __future__ import annotations

import difflib

import click

from sql_agent_cli import config, events, http

__version__ = "0.1.0"


class AskOrCommand(click.Group):
    """A group whose unrecognised first argument is a question, not a typo.

    `sql-agent "how many customers do we have?"` and `sql-agent cache` both have
    to work, so argv is classified before click parses it: the first token that
    is not one of this group's own flags either names a command or begins a
    question, and a question gets `ask` spliced in front of it.

    Note what this rests on — that the group itself has **no value-taking
    options**. A `--url` here would make `sql-agent --url X cache` parse as a
    question, because `X` would be indistinguishable from a command name. That
    is most of why there isn't one.

    The two cases cannot always be told apart. Quoting does not survive `exec`,
    so `sql-agent "cache"` is byte-for-byte `sql-agent cache`. The command wins,
    and `sql-agent ask "cache"` is how to ask a one-word question that collides.
    """

    OWN_FLAGS = frozenset({"-h", "--help", "--version"})

    # Read by `ask` to tell "the user typed a question" from "the user typed
    # `ask`". click shares `meta` down the whole context tree, so the subcommand
    # sees what was set here.
    INFERRED = "sql_agent.inferred_ask"

    def parse_args(self, ctx, args):
        head = next((a for a in args if a not in self.OWN_FLAGS), None)
        if head is not None and head not in self.commands:
            args = ["ask", *args]
            ctx.meta[self.INFERRED] = True
        return super().parse_args(ctx, args)


@click.group(cls=AskOrCommand, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="sql-agent")
def cli() -> None:
    """Ask a database questions in English.

    \b
      sql-agent connect warehouse
      sql-agent "how many customers do we have?"
      sql-agent cache

    The first turn against a database is expensive — it explores. Every turn
    after that reads what it learned, and costs a fraction. `sql-agent cache` is
    what it learned; `sql-agent turns` is what each one cost.

    To ask a one-word question that collides with a command name, spell out
    `sql-agent ask "cache"`.
    """


@cli.command()
@click.argument("question", nargs=-1, required=True)
@config.option
@click.option("-v", "--verbose", is_flag=True, help="Show planning, exploration and what was learned.")
@click.option("--json", "as_json", is_flag=True, help="One raw event per line, unstyled.")
def ask(question, connection, verbose, as_json) -> None:
    """Ask a question. This is what you get by default, so `ask` is optional."""
    if verbose and as_json:
        raise click.UsageError("--json and --verbose are two renderers; pick one")

    joined = " ".join(question)
    inferred = click.get_current_context().meta.get(AskOrCommand.INFERRED, False)
    if (
        inferred
        and " " not in joined
        and (match := difflib.get_close_matches(joined, cli.commands, n=1))
    ):
        # A one-word question that nearly names a command is a typo, and a typo
        # that reaches the model costs a T1's worth of tokens to discover.
        #
        # Only when `ask` was inferred: typing it out is the documented way to
        # ask a question that collides with a command, so guessing there would
        # break the escape hatch this message recommends.
        raise click.UsageError(
            f"no such command {joined!r} — did you mean {match[0]!r}? "
            f"(to ask it as a question: sql-agent ask {joined!r})"
        )

    http.run(_ask(joined, connection, verbose, as_json))


async def _ask(question: str, connection: str | None, verbose: bool, as_json: bool) -> None:
    cid = config.connection(connection)
    if not as_json:
        click.secho(question, fg="yellow", bold=True)

    fatal = False
    # No session_id: the server mints one per turn, which is what a one-shot
    # question wants. Turns share the connection's cache regardless — that is
    # the whole point.
    async for ev in http.stream_events(
        f"/connections/{cid}/ask", {"question": question}
    ):
        if as_json:
            events.raw(ev)
        else:
            events.show(ev, verbose=verbose)
        fatal = fatal or (ev.get("type") == "error" and ev.get("fatal"))

    # A recoverable SQL error carries no `fatal` key — that is the fix loop
    # working, not a failed turn. Exiting 0 on a genuinely failed one is how
    # `make demo` used to record a crash as a good take.
    if fatal:
        raise SystemExit(1)


# Registered here rather than imported at the top: the command modules import
# `config.option`, and `AskOrCommand` needs the full command list before it can
# tell a question from a typo — so the group has to exist first.
from sql_agent_cli import connections as _connections  # noqa: E402
from sql_agent_cli import memory as _memory  # noqa: E402

for _command in (
    _connections.connect,
    _connections.connections,
    _memory.cache,
    _memory.turns,
    _memory.reset,
):
    cli.add_command(_command)


if __name__ == "__main__":  # pragma: no cover
    cli()
