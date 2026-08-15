"""Where the CLI is pointed, and how it decides.

The precedence order is the only part of this the user has to hold in their
head, so it is asserted rather than described. The two error strings are
asserted verbatim: they are the whole user interface for "you haven't set this
up yet", and a reworded one is a support question.
"""

from __future__ import annotations

import json

import pytest

from sql_agent import config
from sql_agent.http import ApiError

NO_CONNECTION = (
    "no connection selected — pick one with 'sql-agent connect <id>' "
    "(see 'sql-agent connections ls')"
)


# ------------------------------------------------------------------ precedence


def test_the_flag_wins_over_everything(monkeypatch):
    monkeypatch.setenv(config.ENV_CONNECTION, "from-env")
    config.select("from-state")
    assert config.connection("from-flag") == "from-flag"


def test_the_environment_beats_the_state_file(monkeypatch):
    """So a CI job or a direnv can retarget one shell without writing to a file
    another process may be reading."""
    monkeypatch.setenv(config.ENV_CONNECTION, "from-env")
    config.select("from-state")
    assert config.connection(None) == "from-env"


def test_the_state_file_is_the_fallback():
    config.select("from-state")
    assert config.connection(None) == "from-state"


def test_with_nothing_set_it_says_exactly_what_to_do():
    with pytest.raises(ApiError) as e:
        config.connection(None)
    assert str(e.value) == NO_CONNECTION


def test_an_unset_url_falls_back_to_what_make_up_starts(monkeypatch):
    """A fresh clone works after `make up` with no setup step, the way every URL
    in app/settings.py does. "no API at ..." is a better first error than "you
    have not configured me"."""
    monkeypatch.delenv(config.ENV_URL)
    assert config.base_url() == "http://localhost:8000/v1"


def test_the_url_keeps_whatever_prefix_it_was_given(monkeypatch):
    monkeypatch.setenv(config.ENV_URL, "  http://elsewhere:3000/v1  ")
    assert config.base_url() == "http://elsewhere:3000/v1"


# ------------------------------------------------------------------ the state


def test_select_writes_a_file_that_selected_reads_back():
    assert config.selected() is None
    config.select("warehouse")
    assert config.selected() == "warehouse"
    assert json.loads(config.state_path().read_text()) == {"connection": "warehouse"}


def test_select_none_clears_it():
    config.select("warehouse")
    config.select(None)
    assert config.selected() is None


def test_a_corrupt_state_file_reads_as_no_selection():
    """A missing file is the normal first run and a corrupt one is not worth a
    traceback: the next `connect` overwrites it, and the message is the same."""
    path = config.state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert config.selected() is None
    with pytest.raises(ApiError, match="no connection selected"):
        config.connection(None)


def test_the_state_file_lives_under_xdg_state_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert config.state_path() == tmp_path / "sql-agent" / "state.json"
