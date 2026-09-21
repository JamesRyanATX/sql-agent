"""OpenRouter: OpenAI's wire shape, with three differences that cost money.

Each test here pins one of them, and each one is silent when it breaks. A
`reasoning_effort` field OpenRouter does not read means every call runs at the
model's default thinking, and the bill arrives at the end of the month. A
missing `cache_control` means the cached turn — which is the whole demo — quietly
costs full price. An unread `cost` means nobody finds out until the statement.

No network: the same mock-transport harness test_llm_openai.py uses.
"""

from __future__ import annotations

import httpx
import pytest

from app import graph, llm
from app.config import Config, Model
from tests.test_llm_openai import OK, capture

ROUTER = "https://openrouter.ai/api/v1"


def use_openrouter(monkeypatch, *, model: str = "google/gemini-2.5-flash") -> Config:
    loaded = Config(model=Model(provider="openrouter", model=model, url=ROUTER))
    monkeypatch.setattr(llm, "config", lambda: loaded)
    return loaded


async def call(monkeypatch, *, effort: str = "high", **kw) -> dict:
    """One request, returning the body that went out."""
    sent = capture(monkeypatch, httpx.Response(200, json=kw.pop("response", OK)))
    await llm.complete(
        system=kw.pop("system", "s"),
        messages=[{"role": "user", "content": "q"}],
        effort=effort,
        max_tokens=kw.pop("max_tokens", 100),
        **kw,
    )
    return sent


# ------------------------------------------------------------------- reasoning


async def test_effort_goes_in_the_reasoning_object(monkeypatch):
    """OpenRouter reads `reasoning`, not the flat field. Sending the flat one
    is not an error — it is ignored, so every call silently runs at the model's
    default thinking and per-node effort stops existing."""
    use_openrouter(monkeypatch)

    sent = await call(monkeypatch, effort="low", schema=graph.PLAN_SCHEMA)

    assert sent["reasoning"] == {"effort": "low"}
    assert "reasoning_effort" not in sent


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh", "max"])
async def test_every_effort_level_survives(monkeypatch, effort):
    """All six, unmapped. The OpenAI path folds `xhigh` and `max` into `high`,
    which would make the top of the ladder a decoration — and the ladder is
    what a config search is searching over."""
    use_openrouter(monkeypatch)

    sent = await call(monkeypatch, effort=effort, schema=graph.PLAN_SCHEMA)

    assert sent["reasoning"]["effort"] == effort


# --------------------------------------------------------------------- caching


async def test_the_system_block_is_marked_for_caching_when_asked(monkeypatch):
    """`graph.plan` asks for this on every turn. Through the plain OpenAI path
    it does nothing at all, which is why a cached turn on that backend costs
    what a cold one does."""
    use_openrouter(monkeypatch, model="anthropic/claude-haiku-4.5")

    sent = await call(
        monkeypatch, system="the cache block", schema=graph.PLAN_SCHEMA, cache_system=True
    )

    assert sent["messages"][0] == {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "the cache block",
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }


async def test_an_unmarked_system_block_stays_a_plain_string(monkeypatch):
    """Every node but `plan`. The block form is only sent where it was asked
    for, because an endpoint that does not know it rejects the request rather
    than ignoring the marker."""
    use_openrouter(monkeypatch)

    sent = await call(monkeypatch, system="plain", schema=graph.PLAN_SCHEMA)

    assert sent["messages"][0] == {"role": "system", "content": "plain"}


# ------------------------------------------------------------------ the money


async def test_it_asks_for_the_charge_and_reads_it_back(monkeypatch):
    """Spend has to be a number on a row, not a surprise on a statement."""
    use_openrouter(monkeypatch)
    priced = {
        **OK,
        "usage": {
            "prompt_tokens": 11_021,
            "completion_tokens": 484,
            "cost": 0.0673,
            "prompt_tokens_details": {"cached_tokens": 9_000},
        },
    }
    sent = capture(monkeypatch, httpx.Response(200, json=priced))

    result = await llm.complete(
        system="s",
        messages=[{"role": "user", "content": "q"}],
        effort="high",
        schema=graph.PLAN_SCHEMA,
        max_tokens=100,
    )

    assert sent["usage"] == {"include": True}
    assert result.cost == 0.0673
    assert result.cache_read == 9_000
    assert result.tokens_in == 11_021


async def test_a_backend_that_says_nothing_reports_no_cost(monkeypatch):
    """`None`, not zero. A turn on a free model costs nothing; a turn on a
    backend that bills elsewhere is unknown, and the turn log must not show
    those as the same thing."""
    use_openrouter(monkeypatch)
    capture(monkeypatch, httpx.Response(200, json=OK))

    result = await llm.complete(
        system="s",
        messages=[{"role": "user", "content": "q"}],
        effort="high",
        schema=graph.PLAN_SCHEMA,
        max_tokens=100,
    )

    assert result.cost is None
    assert result.cache_read == 0


async def test_a_free_model_reports_zero_and_means_it(monkeypatch):
    use_openrouter(monkeypatch, model="openai/gpt-oss-20b:free")
    capture(
        monkeypatch,
        httpx.Response(200, json={**OK, "usage": {**OK["usage"], "cost": 0.0}}),
    )

    result = await llm.complete(
        system="s",
        messages=[{"role": "user", "content": "q"}],
        effort="high",
        schema=graph.PLAN_SCHEMA,
        max_tokens=100,
    )

    assert result.cost == 0.0


# --------------------------------------------------- what it shares with OpenAI


async def test_the_cap_and_the_forced_tool_call_are_unchanged(monkeypatch):
    """Everything else is the OpenAI path. Structured output is still a forced
    tool call, and `max_tokens` is still the cap — which on this backend is also
    the thinking budget, since effort is applied as a fraction of it."""
    use_openrouter(monkeypatch)

    sent = await call(monkeypatch, schema=graph.PLAN_SCHEMA, max_tokens=4_000)

    assert sent["max_tokens"] == 4_000
    assert sent["tool_choice"] == "required"
    assert [t["function"]["name"] for t in sent["tools"]] == [llm._EMIT]


# ------------------------------------------------------------------- the key


def test_openrouter_has_its_own_key_and_falls_back(monkeypatch):
    """Billing is separate, so the name is separate — but one gateway is the
    common case, and two names for one value is a trap."""
    from app.settings import settings

    spec = Model(provider="openrouter", model="x/y", url=ROUTER)

    # Built without the env file, deliberately. `Settings` reads `.env` as well
    # as the environment, so `delenv` alone does not unset anything a developer
    # happens to have in theirs — this used to pass only on a machine with no
    # OpenRouter key, and failed the moment somebody set one up.
    from app.settings import Settings

    monkeypatch.setattr(llm, "settings", lambda: Settings(_env_file=None))

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    settings.cache_clear()
    try:
        assert llm._key_for(spec) == "sk-or-real"

        monkeypatch.delenv("OPENROUTER_API_KEY")
        settings.cache_clear()
        assert llm._key_for(spec) == "sk-openai"
        # And the other provider is unaffected either way.
        assert llm._key_for(Model(provider="openai_compat", model="q", url=ROUTER)) == (
            "sk-openai"
        )
    finally:
        monkeypatch.undo()
        settings.cache_clear()


# ------------------------------------------------------------- rate limiting


async def test_a_rate_limit_is_waited_out_rather_than_raised(monkeypatch):
    """OpenRouter caps a new account at 20 requests a minute per model, and one
    cold turn is five to eight calls — so three concurrent rollouts exceed it in
    seconds. Observed: 17 of 19 rollouts of a paid run died on 429 and the
    tokens already spent bought nothing.

    A 429 costs no tokens, so retrying is free apart from the clock.
    """
    use_openrouter(monkeypatch)
    slept: list[float] = []
    monkeypatch.setattr(llm.asyncio, "sleep", lambda s: slept.append(s) or _noop())

    seen = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] < 3:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=OK)

    monkeypatch.setattr(
        llm, "_http_client", lambda spec: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=ROUTER
        )
    )

    result = await llm.complete(system="s", messages=[], node="plan", effort="low")

    assert seen["n"] == 3, "it kept going until the endpoint stopped refusing"
    assert len(slept) == 2, "and waited between attempts"
    assert result.tool_uses, "and the answer it eventually got is the real one"


async def test_a_rate_limit_that_never_clears_is_reported(monkeypatch):
    """Bounded. A limit that does not clear is a configuration problem, and
    waiting quietly on one is worse than saying so."""
    use_openrouter(monkeypatch)
    monkeypatch.setattr(llm.asyncio, "sleep", lambda s: _noop())
    monkeypatch.setattr(
        llm, "_http_client", lambda spec: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(429, json={"error": {"message": "nope"}})
            ),
            base_url=ROUTER,
        )
    )

    with pytest.raises(llm.LlmError, match="429"):
        await llm.complete(system="s", messages=[], node="plan", effort="low")


def test_the_endpoints_own_delay_is_honoured_over_a_guess():
    """`Retry-After` when there is one, OpenRouter's millisecond reset stamp
    when there is not, and a doubling backoff otherwise."""
    plain = httpx.Response(429, headers={"retry-after": "7"}, json={})
    assert llm._retry_after(plain, 0) == 7.0

    import time as _time

    soon = int((_time.time() + 4) * 1000)
    router = httpx.Response(
        429,
        json={"error": {"metadata": {"headers": {"X-RateLimit-Reset": str(soon)}}}},
    )
    assert 0 < llm._retry_after(router, 0) <= 5

    assert llm._retry_after(httpx.Response(429, json={}), 3) == 8.0


async def _noop():
    return None
