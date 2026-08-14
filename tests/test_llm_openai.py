"""The openai_compat request shape.

Local servers are OpenAI-*shaped*, not OpenAI. Where the spec allows two
spellings of the same instruction, the one a given server implements is a
coin toss — so this file pins the choices that were made for portability rather
than for expressiveness, and the diagnosis that made them findable.

No network: the body is the thing under test, not the model.
"""

from __future__ import annotations

import httpx
import pytest

from app import graph, llm
from app.config import Config, Model


def use_openai(monkeypatch, *, model: str = "qwen-test") -> Config:
    """Point every node at one openai_compat endpoint.

    Which backend a call uses is `config/config.yaml` now, not PROVIDER — and
    `llm` does `from app.config import config`, so the name to patch is the one
    it holds. A whole config file for a request-shape test would be ceremony;
    what these pin is the body, and the body is a function of the spec alone.
    """
    loaded = Config(
        model=Model(provider="openai_compat", model=model, url="http://test/v1")
    )
    monkeypatch.setattr(llm, "config", lambda: loaded)
    return loaded


def capture(monkeypatch, response: httpx.Response) -> dict:
    """Run one `complete()` against a mock transport, returning the body sent."""
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        sent.update(json.loads(request.content))
        return response

    # Takes the spec `complete()` resolved from the node, and ignores it: what
    # is under test is the body, and the address it would have gone to is the
    # one thing a MockTransport cannot honour anyway.
    monkeypatch.setattr(
        llm,
        "_http_client",
        lambda spec: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test/v1"
        ),
    )
    return sent


OK = {
    "choices": [
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "function": {"name": llm._EMIT, "arguments": '{"sufficient": false}'},
                    }
                ],
            }
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}


async def test_a_forced_schema_call_uses_the_string_tool_choice(monkeypatch):
    """`"required"`, not `{"type": "function", "function": {...}}`.

    They mean the same thing here — the tools list holds exactly one entry, so
    "call some tool" and "call `emit`" are the same instruction — but only one
    of the two is universally implemented. LM Studio answers the named form with
    `400 Invalid tool_choice type: 'object'`, which is how this was found.
    """
    use_openai(monkeypatch)
    sent = capture(monkeypatch, httpx.Response(200, json=OK))
    await llm.complete(
        system="s", messages=[{"role": "user", "content": "q"}],
        schema=graph.PLAN_SCHEMA, effort="low", max_tokens=100,
    )
    assert sent["tool_choice"] == "required"
    # And the equivalence it rests on: exactly one tool is offered, so
    # "required" cannot select anything else.
    assert [t["function"]["name"] for t in sent["tools"]] == [llm._EMIT]


async def test_ordinary_tool_calls_do_not_force_anything(monkeypatch):
    """`explore` offers four tools and must be free to answer without calling
    one — that is how the loop ends."""
    use_openai(monkeypatch)
    sent = capture(monkeypatch, httpx.Response(200, json={
        "choices": [{"message": {"content": "hi"}}], "usage": {},
    }))
    from app.tools import SCHEMAS

    await llm.complete(
        system="s", messages=[{"role": "user", "content": "q"}],
        tools=SCHEMAS, effort="high", max_tokens=100,
    )
    assert "tool_choice" not in sent
    assert len(sent["tools"]) == len(SCHEMAS)


async def test_a_rejected_request_reports_what_the_server_said(monkeypatch):
    """`raise_for_status()` reports the status and the URL and throws the body
    away — and the body is the entire diagnosis. "400 Bad Request for url ..."
    plus a link to MDN sends you looking at the network instead of at the one
    field the server did not like."""
    use_openai(monkeypatch)
    capture(monkeypatch, httpx.Response(
        400, json={"error": "Invalid tool_choice type: 'object'."}
    ))
    with pytest.raises(llm.LlmError) as e:
        await llm.complete(
            system="s", messages=[{"role": "user", "content": "q"}],
            effort="low", max_tokens=100,
        )
    assert "Invalid tool_choice" in str(e.value)
    assert "400" in str(e.value)
