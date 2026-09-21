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

from sql_agent import config, http, turn

__version__ = "0.1.0"


class AskOrCommand(click.Group):
    """A group whose unrecognised first argument is a question, not a typo.

    argv is classified before click parses it: the first token that is not one
    of this group's own flags either names a command or begins a question, and a
    question gets `ask` spliced in front of it.

    This rests on the group having **no value-taking options** — a `--url` here
    would make `sql-agent --url X cache` parse as a question.

    Quoting does not survive `exec`, so `sql-agent "cache"` is byte-for-byte
    `sql-agent cache`. The command wins, and `sql-agent ask "cache"` is how to
    ask a one-word question that collides.
    """

    OWN_FLAGS = frozenset({"-h", "--help", "--version"})

    # Read by `ask` to tell a typed question from a typed `ask`. click shares
    # `meta` down the context tree, so the subcommand sees this.
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
      sql-agent "how many customers do we have?"
      sql-agent cache
      sql-agent turns

    The first turn against a database is expensive — it explores. Every turn
    after that reads what it learned, and costs a fraction. `sql-agent cache` is
    what it learned; `sql-agent turns` is what each one cost.

    To ask a one-word question that collides with a command name, spell out
    `sql-agent ask "cache"`.
    """


@cli.command()
@click.argument("question", nargs=-1, required=True)
@click.option("-v", "--verbose", is_flag=True, help="Show planning, exploration and what was learned.")
@click.option("--json", "as_json", is_flag=True, help="One raw event per line, unstyled.")
@click.option(
    "--no-feedback",
    is_flag=True,
    help="Don't ask what you thought of the answer.",
)
@click.option(
    "--no-memory",
    is_flag=True,
    help="Ignore what the agent has learned, and keep nothing it learns.",
)
def ask(question, verbose, as_json, no_feedback, no_memory) -> None:
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
        # reaching the model costs a T1's worth of tokens to discover. Only when
        # `ask` was inferred, or this would break the escape hatch it recommends.
        raise click.UsageError(
            f"no such command {joined!r} — did you mean {match[0]!r}? "
            f"(to ask it as a question: sql-agent ask {joined!r})"
        )

    http.run(_ask(joined, verbose, as_json, no_feedback, no_memory))


async def _ask(
    question: str,
    verbose: bool,
    as_json: bool,
    no_feedback: bool = False,
    no_memory: bool = False,
) -> None:
    answered, fatal = await turn.take(
        question, verbose=verbose, as_json=as_json, memory=not no_memory
    )

    # A recoverable SQL error carries no `fatal` key — that is the fix loop
    # working, not a failed turn. Exiting 0 on a genuinely failed one would let
    # `make demo` record a crash as a good take.
    if fatal:
        raise SystemExit(1)

    if not no_feedback and not as_json and turn.askable(answered):
        await turn.judge(answered)


# Registered here rather than imported at the top: the command modules import
# `config.option`, so the group has to exist first.
from sql_agent import corpus as _corpus  # noqa: E402
from sql_agent import memory as _memory  # noqa: E402
from sql_agent import server as _server  # noqa: E402

for _command in (
    _corpus.record,
    _memory.cache,
    _memory.turns,
    _memory.reset,
    _server.show_config,
):
    cli.add_command(_command)


if __name__ == "__main__":  # pragma: no cover
    cli()
