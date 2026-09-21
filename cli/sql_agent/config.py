"""Where the CLI is pointed.

Two environment variables, and both have working defaults. Nothing under
`cli/sql_agent/` imports `app` — the CLI is an HTTP client of the server, not a
second way into it — so these names are the CLI's own rather than the server's,
and the 401 message names both sides.
"""

from __future__ import annotations

import os

ENV_URL = "SQL_AGENT_URL"
ENV_KEY = "SQL_AGENT_API_KEY"

# What `make up` starts. Defaulted rather than required so a fresh clone works
# with no setup step, and because "no API at http://localhost:8000/v1" is a
# better first error than "you have not configured me".
DEFAULT_URL = "http://localhost:8000/v1"


def base_url() -> str:
    """The server, including its `/v1` prefix.

    httpx keeps a base_url's path when it merges a relative one, so a call site
    writes `/cache` and gets `/v1/cache`.
    """
    return os.environ.get(ENV_URL, "").strip() or DEFAULT_URL


def api_key() -> str:
    return os.environ.get(ENV_KEY, "")
