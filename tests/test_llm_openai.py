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
from app.settings import settings


def capture(monkeypatch, response: httpx.Response) -> dict:
    """Run one `complete()` against a mock transport, returning the body sent."""
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        sent.update(json.loads(request.content))
        return response

    monkeypatch.setattr(
        llm,
        "_http_client",
        lambda: httpx.AsyncClient(
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
    monkeypatch.setenv("PROVIDER", "openai_compat")
    settings.cache_clear()
    try:
        sent = capture(monkeypatch, httpx.Response(200, json=OK))
        await llm.complete(
            system="s", messages=[{"role": "user", "content": "q"}],
            schema=graph.PLAN_SCHEMA, effort="low", max_tokens=100,
        )
        assert sent["tool_choice"] == "required"
        # And the equivalence it rests on: exactly one tool is offered, so
        # "required" cannot select anything else.
        assert [t["function"]["name"] for t in sent["tools"]] == [llm._EMIT]
    finally:
        settings.cache_clear()


async def test_ordinary_tool_calls_do_not_force_anything(monkeypatch):
    """`explore` offers four tools and must be free to answer without calling
    one — that is how the loop ends."""
    monkeypatch.setenv("PROVIDER", "openai_compat")
    settings.cache_clear()
    try:
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
    finally:
        settings.cache_clear()


async def test_a_rejected_request_reports_what_the_server_said(monkeypatch):
    """`raise_for_status()` reports the status and the URL and throws the body
    away — and the body is the entire diagnosis. "400 Bad Request for url ..."
    plus a link to MDN sends you looking at the network instead of at the one
    field the server did not like."""
    monkeypatch.setenv("PROVIDER", "openai_compat")
    settings.cache_clear()
    try:
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
    finally:
        settings.cache_clear()
