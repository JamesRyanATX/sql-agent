"""Prompt evaluation and search. A development tool, not part of the server.

`optim/` reads the app; the app never reads `optim/`, which
[tests/test_cli_isolation.py](tests/test_cli_isolation.py) asserts. It is in the
same category as pytest: it imports `app` directly rather than going through the
API, because it operates on the agent's source rather than being a client of a
running one. The boundary that does hold: **node replay is in-process, turn
replay is HTTP** — re-driving `stream_turn` by hand is how a harness and a
product drift, and `POST /v1/connections/{id}/ask` is the path that ships.

`gepa` lives in the `optim` dependency group and is imported by `adapter.py`
alone, so the probe suite and the metric run with nothing extra installed.
"""
