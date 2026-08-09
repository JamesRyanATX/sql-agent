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

from sql_agent_cli import events

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


def test_a_default_turn_is_two_lines():
    out = render({"type": "rows", "count": 1, "rows": [{"n": 1840}]}, ANSWER)
    assert out == "=> [(1840)]\n\n405 tokens (215 in / 190 out) in 6.6s  [no exploration]\n"


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
