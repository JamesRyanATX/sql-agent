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
