"""GEPA over the agent's prompts. A development tool, not part of the server.

It reads the app; the app never reads it (tests/test_cli_isolation.py). Node
replay is in-process, turn replay is HTTP — re-driving `stream_turn` by hand is
how a harness and a product drift.

Nested one level because a top-level `gepa/` would shadow the library it wraps:
`sys.path[0]` is the repository root, so it would beat site-packages and
`from gepa.core.adapter import GEPAAdapter` would fail. Do not flatten it.

`gepa` is a dependency group, imported by `adapter.py` alone.
"""
