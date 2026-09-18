"""What the server is running. Not about any one database, so no -c."""

from __future__ import annotations

import click

from sql_agent import http, render


@click.command("config")  # explicit: the function name would become `show-config`
def show_config() -> None:
    """The model configuration the server is running.

    config.yaml with config.local.yaml merged over it, shaped like the file.
    Every line the overlay decided is marked, so what is answering questions
    is never a surprise.
    """
    http.run(_config())


async def _config() -> None:
    body = await http.get("/config")
    if body["overlay"]:
        click.echo(render.dim(f"# {body['overlay']} is overlaying config.yaml"))
    render.tree(
        body["config"],
        marked=frozenset(body["overridden"]),
        note="# config.local.yaml",
    )
    # Last, and outside the tree, because it is not in either file — it is the
    # two Langfuse keys in the server's environment. Printed either way: off is
    # the answer to "why did my verdict not land", and silence is not.
    state = "on" if body["tracing"] else "off"
    click.echo(render.dim(f"\n# tracing is {state}"))
