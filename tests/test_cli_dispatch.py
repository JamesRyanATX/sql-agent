"""Telling a question from a command.

`sql-agent cache` and `sql-agent "how many customers do we have?"` are the same
shape to a process, so `AskOrCommand` classifies argv before click parses it.
This file is the table of what that must do — including the two cases it
deliberately gets "wrong", because quoting does not survive `exec`.

Plain `def`, not `async def`: every command ends in `http.run(coro)` →
`asyncio.run`, and CliRunner is synchronous. A test that also had a running loop
would be driving two.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from sql_agent.main import cli


@pytest.fixture
def spy(monkeypatch):
    """Replace the network half of every command, and record what it was given."""
    seen: dict = {}

    def record(name):
        async def fake(*args, **kwargs):
            seen.update(command=name, args=args, kwargs=kwargs)

        return fake

    from sql_agent import main, memory, server

    monkeypatch.setattr(main, "_ask", record("ask"))
    monkeypatch.setattr(memory, "_cache", record("cache"))
    monkeypatch.setattr(memory, "_turns", record("turns"))
    monkeypatch.setattr(memory, "_reset", record("reset"))
    monkeypatch.setattr(server, "_config", record("config"))
    return seen


def invoke(argv: list[str]) -> tuple[int, str]:
    result = CliRunner().invoke(cli, argv)
    return result.exit_code, result.output


# ------------------------------------------------------- a question, or a command


def test_a_subcommand_is_a_subcommand(spy):
    assert invoke(["cache"])[0] == 0
    assert spy["command"] == "cache"


def test_a_subcommand_keeps_its_own_flags(spy):
    assert invoke(["cache", "--kind", "recipe"])[0] == 0
    assert spy == {"command": "cache", "args": ("recipe",), "kwargs": {}}


def test_a_quoted_question_arrives_intact(spy):
    assert invoke(["how many customers do we have?"])[0] == 0
    assert spy["args"][0] == "how many customers do we have?"


def test_an_unquoted_question_is_joined_with_spaces(spy):
    """`" ".join(sys.argv[1:])` is what scripts/ask.py did, and people type it."""
    assert invoke(["how", "many", "customers"])[0] == 0
    assert spy["args"][0] == "how many customers"


@pytest.mark.parametrize(
    "argv",
    [["-v", "how many customers?"], ["how many customers?", "-v"]],
)
def test_a_flag_works_on_either_side_of_the_question(spy, argv):
    """The leading-option form is the one that would break a naive
    `args[0] not in commands` rule."""
    assert invoke(argv)[0] == 0
    assert spy["command"] == "ask"
    assert spy["args"] == ("how many customers?", True, False, False)


def test_ask_is_how_you_ask_a_question_that_is_a_command(spy):
    """Same string, two meanings, and the only way to pick the other one."""
    assert invoke(["ask", "cache"])[0] == 0
    # question, verbose, as_json, no_memory
    assert spy == {
        "command": "ask",
        "args": ("cache", False, False, False),
        "kwargs": {},
    }

    assert invoke(["cache"])[0] == 0
    assert spy["command"] == "cache"


@pytest.mark.parametrize("typo, meant", [("cahce", "cache"), ("confg", "config")])
def test_a_near_miss_for_a_command_is_a_typo_not_a_question(spy, typo, meant):
    """A question costs a model call and minutes of wall clock, so discovering
    `cahce` that way is expensive. Multi-word questions are never guessed at."""
    code, output = invoke([typo])
    assert code == 2
    assert f"did you mean '{meant}'" in output
    assert f"sql-agent ask '{typo}'" in output
    assert spy == {}


def test_config_is_a_command_not_a_question(spy):
    """It reports the server's own settings, so it takes no options at all."""
    assert invoke(["config"])[0] == 0
    assert spy == {"command": "config", "args": (), "kwargs": {}}
    assert invoke(["config", "--kind", "recipe"])[0] == 2


def test_an_unambiguous_one_word_question_is_still_asked(spy):
    assert invoke(["revenue"])[0] == 0
    assert spy["command"] == "ask"


def test_verbose_and_json_are_two_renderers(spy):
    code, output = invoke(["-v", "--json", "how many?"])
    assert code == 2
    assert "pick one" in output


# ------------------------------------------------------------------- the group


@pytest.mark.parametrize("argv", [["--help"], ["-h"], ["--version"]])
def test_the_groups_own_flags_still_reach_click(argv):
    assert invoke(argv)[0] == 0


def test_a_bare_invocation_prints_help():
    code, output = invoke([])
    assert code == 2
    assert "Ask a database questions" in output and "corpus" in output


def test_an_unknown_option_is_still_an_unknown_option():
    code, output = invoke(["--nope"])
    assert code == 2
    assert "No such option" in output


def test_no_memory_is_a_flag_on_ask_and_nothing_else(spy):
    """The optimiser's one question-shaped need. It has to reach `_ask` as a
    value, because a flag that parsed and did nothing would still exit 0."""
    assert invoke(["--no-memory", "how many customers?"])[0] == 0
    assert spy["args"] == ("how many customers?", False, False, True)

    assert invoke(["cache", "--no-memory"])[0] == 2
