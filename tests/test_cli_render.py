"""What a turn looks like on the way past.

The default output is a contract with `demo/demo.tape`, not a preference. VHS
records a fixed terminal and can only see one screenful; it waits on
`[explored]`, `[no exploration]` and `tokens (`. An extra line printed by
default scrolls the awaited one out of view, and every recording after that
times out at forty minutes. So the silence below is asserted, not assumed.
"""

from __future__ import annotations

import json

import click
import pytest
from click.testing import CliRunner

from sql_agent import events
from sql_agent import render as render_mod

ANSWER = {
    "type": "answer",
    "text": "1,840 active customers.",
    "sql": "SELECT count(*) FROM customer WHERE deleted_at IS NULL",
    "tokens_in": 215,
    "tokens_out": 190,
    "total_tokens": 405,
    "latency_ms": 6600,
    "explored": False,
}


def render(*evs, verbose: bool = False) -> str:
    """Run the renderer and capture it, colour stripped as a pipe would."""
    runner = CliRunner()

    @click.command()
    def show():
        for ev in evs:
            events.show(ev, verbose=verbose)

    result = runner.invoke(show, color=False)
    assert result.exception is None, result.exception
    return result.output


# --------------------------------------------------------------- the cost line


def test_the_cost_line_is_a_golden_string():
    """demo/demo.tape waits on `tokens (`. Reformatting this breaks the take."""
    assert events.cost(ANSWER) == (
        "405 tokens (215 in / 190 out) in 6.6s  [no exploration]"
    )


def test_an_explored_turn_says_so():
    assert events.cost({**ANSWER, "explored": True}).endswith("  [explored]")


def test_thousands_are_separated():
    """T1 is five figures, and 11726 does not read as a number at a glance."""
    line = events.cost({**ANSWER, "total_tokens": 11_726, "tokens_in": 11_021})
    assert "11,726 tokens (11,021 in" in line


# ------------------------------------------------------------- what shows when


@pytest.mark.parametrize(
    "ev",
    [
        {"type": "plan", "cache_entries": 5, "sufficient": True, "used": ["x"], "missing": []},
        {"type": "explore", "tool": "list_tables", "input": {}, "error": False, "calls": 1},
        {"type": "findings", "text": "customer uses soft deletes", "tool_calls": 5},
        {"type": "sql", "sql": "SELECT 1", "assumptions": []},
        {"type": "learned", "count": 2, "skipped": 0, "entries": []},
        {"type": "usage", "node": "plan", "tokens_in": 10, "tokens_out": 2},
        {"type": "done"},
    ],
)
def test_nothing_but_rows_errors_and_answers_prints_by_default(ev):
    """The tape's line budget, asserted one event type at a time."""
    assert render(ev) == ""


def test_a_default_turn_is_a_table_and_a_cost_line():
    """The whole default view, as a golden string: what was asked for, and what
    it cost. Nothing that explains either.

    A one-cell result gets the full table anyway, because psql does not
    special-case one and neither should this — and because a `=>` shorthand for
    the scalar case is a second output format to keep working.
    """
    out = render({"type": "rows", "count": 1, "rows": [{"customer_count": 1840}]}, ANSWER)
    assert out == (
        "\n"
        "customer_count\n"
        "--------------\n"
        "          1840\n"
        "\n"
        "(1 row)\n"
        "\n"
        "405 tokens (215 in / 190 out) in 6.6s  [no exploration]\n"
    )


def test_columns_align_by_what_is_in_them_not_by_their_name():
    """`table`'s `right=` allowlist is by header name, which cannot work here:
    the columns are whatever SQL the model just wrote. The Decimal is the case
    that matters — app/graph.py's JSON round trip hands it over as a string, so
    an isinstance check alone would leave it left-aligned beside its own kind."""
    out = render(
        {
            "type": "rows",
            "count": 2,
            "rows": [
                {"region": "west", "orders": 408, "revenue": "1234.50"},
                {"region": "east", "orders": 736, "revenue": "98.75"},
            ],
        }
    )
    assert "region | orders | revenue" in out
    assert "west   |    408 | 1234.50" in out
    assert "east   |    736 |   98.75" in out


def test_a_null_is_blank_not_the_word_none():
    out = render({"type": "rows", "count": 1, "rows": [{"a": 1, "b": None}]})
    assert "None" not in out
    assert "\n1 |\n" in out


def test_a_long_value_is_truncated_rather_than_wrapped():
    """One long cell must not destroy the alignment of every row below it, or
    push the tape's awaited line off the one screenful VHS can see. `--json` is
    the untruncated view."""
    long = "x" * 200
    out = render({"type": "rows", "count": 1, "rows": [{"note": long}]})
    assert long not in out
    assert "…" in out
    assert max(len(line) for line in out.splitlines()) <= render_mod.CELL


def test_a_full_page_says_more_may_exist_rather_than_a_total_it_lacks():
    """`execute` uses fetchmany(max_rows), so a full page and a result that
    happened to be exactly that long are the same thing on the wire."""
    rows = [{"n": i} for i in range(50)]
    assert "(50 rows — more matched)" in render(
        {"type": "rows", "count": 50, "rows": rows, "capped": True}
    )
    assert "more matched" not in render(
        {"type": "rows", "count": 50, "rows": rows, "capped": False}
    )


def test_an_empty_result_prints_a_footer_and_no_header():
    """No row means no keys, so there is no header available to print."""
    assert render({"type": "rows", "count": 0, "rows": []}) == "(0 rows)\n"


def test_the_answer_text_is_verbose_only():
    """Asserted in both directions, because this has now moved twice.

    The prose explains a result the reader is already looking at, so it sits with
    the other explanations behind -v — but it must still be reachable, and the
    cost line must not follow it there."""
    assert "1,840 active customers." not in render(ANSWER)
    assert "1,840 active customers." in render(ANSWER, verbose=True)
    assert "405 tokens" in render(ANSWER)


def test_a_turn_with_no_answer_text_still_prints_its_cost():
    out = render({**ANSWER, "text": ""})
    assert out == "\n405 tokens (215 in / 190 out) in 6.6s  [no exploration]\n"


def test_verbose_shows_the_work():
    out = render(
        {"type": "plan", "cache_entries": 5, "sufficient": True, "used": ["revenue"], "missing": []},
        {"type": "explore", "tool": "describe_table", "input": {"name": "customer"},
         "error": False, "calls": 3},
        {"type": "sql", "sql": "SELECT 1", "assumptions": ["orders.created, not created_at"]},
        verbose=True,
    )
    assert "plan: 5 entries; sufficient; used revenue" in out
    assert "3. describe_table(name='customer')" in out
    assert "assumes: orders.created" in out


def test_a_cold_cache_plan_has_no_used_key(monkeypatch):
    """The short-circuit emits no `used` at all — the plan node never ran, so
    there is nothing it could have used. `.get`, not `[]`."""
    out = render(
        {"type": "plan", "cache_entries": 0, "sufficient": False, "missing": []},
        verbose=True,
    )
    assert "plan: 0 entries; insufficient" in out
    assert "used" not in out


@pytest.mark.parametrize(
    "ev,expected",
    [
        ({"type": "learned", "count": 0, "skipped": 0, "entries": [], "cached": True},
         "the cache already covered it"),
        ({"type": "learned", "count": 0, "skipped": 0, "entries": [],
          "failed": "TimeoutError: took too long"},
         "extraction failed — TimeoutError"),
        ({"type": "learned", "count": 1, "skipped": 1,
          "entries": [{"name": "revenue", "kind": "recipe", "claim": "excludes cancelled",
                       "verified": True}]},
         "learned: 1 entries, 1 skipped"),
    ],
)
def test_all_three_learned_shapes_render(ev, expected):
    """One event name, three payloads — a renderer that assumed one KeyErrors on
    the other two, mid-turn, after the tokens are already spent."""
    assert expected in render(ev, verbose=True)


def test_an_explore_error_is_marked():
    out = render(
        {"type": "explore", "tool": "describe_table", "input": {"name": "nope"},
         "error": True, "calls": 2},
        verbose=True,
    )
    assert out.startswith("!")


# --------------------------------------------------------------------- --json


def test_json_emits_one_parseable_object_per_event():
    runner = CliRunner()

    @click.command()
    def show():
        for ev in ({"type": "usage", "node": "plan"}, ANSWER, {"type": "done"}):
            events.raw(ev)

    out = runner.invoke(show, color=False).output
    parsed = [json.loads(line) for line in out.splitlines()]
    # `usage` is dropped by every other renderer and kept here — raw is raw.
    assert [p["type"] for p in parsed] == ["usage", "answer", "done"]


# ----------------------------------------------------------------------- tree


def tree(data: dict, marked: frozenset[str] = frozenset()) -> str:
    runner = CliRunner()

    @click.command()
    def show():
        render_mod.tree(data, marked=marked, note="# local")

    result = runner.invoke(show, color=False)
    assert result.exception is None, result.exception
    return result.output


def test_tree_prints_the_file_the_user_knows():
    """`sql-agent config` output is the yaml the user edits: key order kept,
    unset nodes absent, `900.0` back to the `900` the file says."""
    out = tree(
        {
            "model": {"provider": "openai_compat", "url": "http://h/v1", "timeout": 900.0},
            "max_rows": 50,
            "plan": {"effort": None, "model": None},
            "explore": {"effort": "low", "model": None},
            "flag": True,
        },
        marked=frozenset({"model.provider", "explore.effort"}),
    )
    assert out == (
        "model:\n"
        "  provider: openai_compat  # local\n"
        "  url: http://h/v1\n"
        "  timeout: 900\n"
        "max_rows: 50\n"
        "explore:\n"
        "  effort: low  # local\n"
        "flag: true\n"
    )


def test_tree_shows_a_null_the_overlay_set_on_purpose():
    """Unset is silence; set-to-null by the overlay is a fact, and gets a line.
    A block the overlay replaced wholesale carries the mark on its header."""
    out = tree(
        {"plan": {"effort": None, "model": None}, "explore": {"effort": "high", "model": None}},
        marked=frozenset({"plan.effort", "explore"}),
    )
    assert out == (
        "plan:\n"
        "  effort: null  # local\n"
        "explore:  # local\n"
        "  effort: high\n"
    )


# --------------------------------------------------------------------- choose


def scripted(*presses: str):
    """A reader handing back whole keypresses, the way a terminal does.

    `click.getchar` reads up to 32 bytes at once on a terminal, so one arrow
    arrives as `"\x1b[A"`. Under CliRunner the same press arrives one character
    at a time. Both shapes are tested, because a menu that assumes either one
    works in exactly one of the two places.
    """
    it = iter(presses)
    return lambda: next(it)


def menu(reader) -> tuple[int, str]:
    runner = CliRunner()
    chosen: list[int] = []

    @click.command()
    def show():
        chosen.append(render_mod.choose("Was that right?", ("OK", "Not OK"), read=reader))

    result = runner.invoke(show, color=False)
    assert result.exception is None, result.exception
    return chosen[0], result.output


def test_enter_takes_the_highlighted_option():
    chosen, out = menu(scripted("\r"))
    assert chosen == 0
    # Both options are on screen before anything is pressed, so the second one
    # is discoverable without knowing it is there.
    assert "1. OK" in out and "2. Not OK" in out


def test_an_arrow_moves_the_highlight():
    """The whole-sequence shape, which is what a real terminal sends."""
    assert menu(scripted("\x1b[B", "\r"))[0] == 1


def test_an_arrow_split_across_reads_is_still_one_arrow():
    """CliRunner's stdin is read a character at a time (click/testing.py), so
    the same press arrives in three pieces. Reassembled, or the tests pass
    while the menu is unusable by hand."""
    assert menu(scripted("\x1b", "[", "B", "\r"))[0] == 1


def test_it_wraps_at_the_ends():
    """Two options and a wrap means up and down both reach the other one,
    which is what a two-item menu should do."""
    assert menu(scripted("\x1b[A", "\r"))[0] == 1
    assert menu(scripted("\x1b[B", "\x1b[B", "\r"))[0] == 0


def test_vim_keys_move_too():
    assert menu(scripted("j", "\r"))[0] == 1
    assert menu(scripted("j", "k", "\r"))[0] == 0


def test_a_digit_takes_that_option_outright():
    """The list is numbered, so anyone who reads "2. Not OK" and types 2 is
    right. No Enter — the number is the whole answer."""
    assert menu(scripted("2"))[0] == 1
    assert menu(scripted("1"))[0] == 0


def test_a_key_that_means_nothing_is_ignored():
    assert menu(scripted("x", "9", " ", "2"))[0] == 1
