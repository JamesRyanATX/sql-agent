"""The one place that talks to a model (PLAN.md §7.1).

Two backends behind one `complete()`:

- **anthropic** — `claude-opus-5`, the demo model. Adaptive thinking, effort,
  native structured outputs.
- **openai_compat** — any OpenAI-shaped endpoint, including Ollama. For local
  development when the Anthropic key isn't available.

Which of the two a call uses is `config/config.yaml`, resolved per `node=`: a
global `model:` block, and an optional override on any node. Nodes never see the
difference, but they must build messages with `assistant_turn()` and
`tool_results()` rather than literal dicts — the two wire formats disagree about
tool results, and both take a `node` because an override can change the backend.

Structured output also differs: Anthropic constrains the response directly, while
an OpenAI-compatible endpoint gets a single forced tool call whose parameters are
the schema. The tool channel is JSON-shaped by construction, so it works where
grammar-constrained decoding is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx

from app import tracing
from app.config import Model, config
from app.settings import settings

log = logging.getLogger(__name__)

# On Opus 5 max_tokens caps thinking *plus* response text, and thinking is on by
# default — sized tight, answers truncate mid-sentence. 16k also keeps
# non-streaming requests inside the SDK's HTTP timeout.
MAX_TOKENS = 16_000

# The name of the synthetic tool used to carry structured output on
# openai_compat. Never collides with a real tool — schema calls pass no tools.
_EMIT = "respond"

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

_anthropic_client: anthropic.AsyncAnthropic | None = None
# Keyed by (url, timeout): a per-node override may name another endpoint, and
# httpx binds both to the client. Two nodes on one endpoint share a pool.
_http: dict[tuple[str, float], httpx.AsyncClient] = {}


class Refusal(Exception):
    """The model declined. HTTP 200, empty or partial content — not an API error."""

    def __init__(self, category: str | None, explanation: str | None):
        self.category = category
        super().__init__(f"model refused ({category or 'unspecified'}): {explanation}")


@dataclass(slots=True)
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]


class LlmError(Exception):
    """A model server said no, with the reason it gave.

    `stream_turn` renders this as one fatal error event, so the message is what
    a user sees — hence the server's own body goes in it.
    """


@dataclass(slots=True)
class Result:
    text: str = ""
    tool_uses: list[ToolUse] = field(default_factory=list)
    stop_reason: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read: int = 0
    # What this call charged, in dollars, when the backend says so. None where
    # nothing reported it, which is not the same as free — an Anthropic-direct
    # call costs money and simply does not itemise it in the response.
    cost: float | None = None
    raw: Any = None  # provider-native assistant content, for echo-back

    def parsed(self) -> dict[str, Any]:
        """The JSON a structured-output call produced."""
        return json.loads(self.text)


# ------------------------------------------------------------------- message API


def assistant_turn(result: Result, *, node: str) -> dict[str, Any]:
    """The assistant turn to append before handing back tool results.

    `node` decides the backend, and the two disagree about this shape.
    """
    if config().model_for(node).provider == "anthropic":
        return {"role": "assistant", "content": result.raw}
    return {
        "role": "assistant",
        "content": result.text or None,
        "tool_calls": [
            {
                "id": t.id,
                "type": "function",
                "function": {"name": t.name, "arguments": json.dumps(t.input)},
            }
            for t in result.tool_uses
        ],
    }


def tool_results(
    items: list[tuple[str, str, bool]], *, node: str
) -> list[dict[str, Any]]:
    """Tool outputs as messages. `items` is (tool_use_id, payload, is_error).

    Anthropic wants every result in a *single* user message — splitting them
    trains the model out of calling tools in parallel. OpenAI wants one `tool`
    message each. `node` picks the backend, as in `assistant_turn()`.
    """
    if config().model_for(node).provider == "anthropic":
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": payload,
                        "is_error": is_error,
                    }
                    for tid, payload, is_error in items
                ],
            }
        ]
    return [
        {"role": "tool", "tool_call_id": tid, "content": payload}
        for tid, payload, _ in items
    ]


# --------------------------------------------------------------------- anthropic


def _anthropic() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        key = settings().anthropic_api_key
        _anthropic_client = (
            anthropic.AsyncAnthropic(api_key=key) if key else anthropic.AsyncAnthropic()
        )
    return _anthropic_client


def _strip_pre_fallback(content: list[Any]) -> list[Any]:
    """Make an assistant turn safe to echo back after a mid-output fallback:
    blocks from before the boundary belong to a model that is no longer
    answering, and replaying them is rejected.
    """
    last = max(
        (i for i, b in enumerate(content) if getattr(b, "type", None) == "fallback"),
        default=None,
    )
    if last is None:
        return content
    drop = {"thinking", "redacted_thinking", "tool_use"}
    return [
        b for i, b in enumerate(content) if i > last or getattr(b, "type", None) not in drop
    ]


async def _complete_anthropic(
    *,
    spec: Model,
    system: str,
    messages: list[dict[str, Any]],
    effort: str,
    tools: list[dict[str, Any]] | None,
    schema: dict[str, Any] | None,
    max_tokens: int,
    cache_system: bool,
) -> Result:
    output_config: dict[str, Any] = {"effort": effort}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}

    # The cached path re-sends the same system prompt and the whole cache every
    # turn. The breakpoint goes on the system block because it is the stable
    # prefix; the question after it is the only part that varies.
    system_param: Any = system
    if cache_system:
        system_param = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]

    kwargs: dict[str, Any] = {
        "model": spec.model,
        "max_tokens": max_tokens,
        "system": system_param,
        "messages": messages,
        "output_config": output_config,
    }
    if tools:
        kwargs["tools"] = tools

    # Thinking is left unset, which runs adaptive on Opus 5. Cost is controlled
    # with effort, never by disabling thinking: with it off the model can write a
    # tool call into visible text, and in the explore loop it would never run.
    if settings().use_fallbacks:
        resp = await _anthropic().beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
        )
    else:
        resp = await _anthropic().messages.create(**kwargs)

    # Before touching content: a refusal is a successful HTTP response whose
    # content is empty or partial.
    if resp.stop_reason == "refusal":
        details = getattr(resp, "stop_details", None)
        raise Refusal(
            getattr(details, "category", None), getattr(details, "explanation", None)
        )

    return Result(
        text="".join(b.text for b in resp.content if b.type == "text"),
        tool_uses=[
            ToolUse(id=b.id, name=b.name, input=dict(b.input))
            for b in resp.content
            if b.type == "tool_use"
        ],
        stop_reason=resp.stop_reason,
        tokens_in=resp.usage.input_tokens,
        tokens_out=resp.usage.output_tokens,
        cache_read=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        raw=_strip_pre_fallback(list(resp.content)),
    )


# ----------------------------------------------------------------- openai_compat


def _http_client(spec: Model) -> httpx.AsyncClient:
    """One client per endpoint. The key is in env; the address is in the yaml."""
    key = (spec.url or "", spec.timeout)
    client = _http.get(key)
    if client is None:
        headers = {"authorization": f"Bearer {_key_for(spec)}"}
        if spec.provider == "openrouter":
            # Optional, and worth sending: OpenRouter groups spend by app, which
            # is the difference between "$4 last week" and "$4 on the corpus run
            # last week".
            headers["http-referer"] = "https://github.com/sql-agent"
            headers["x-title"] = "sql-agent"
        client = httpx.AsyncClient(
            base_url=spec.url or "",
            # A 27B model on consumer hardware takes minutes per call, and the
            # default 5s read timeout would fail every request.
            timeout=httpx.Timeout(spec.timeout, connect=10.0),
            headers=headers,
        )
        _http[key] = client
    return client


def _key_for(spec: Model) -> str:
    """Which key dials this endpoint.

    OpenRouter bills separately, so it has its own name and falls back to the
    OpenAI one — which keeps a single-gateway setup to one variable, without
    making somebody overwrite a key they still use for something else.
    """
    s = settings()
    if spec.provider == "openrouter":
        return s.openrouter_api_key or s.openai_api_key
    return s.openai_api_key


# Which spelling of the output cap an endpoint takes, keyed by url. OpenAI's
# reasoning models reject `max_tokens` with a 400 that names
# `max_completion_tokens`; Ollama's OpenAI layer knows only `max_tokens` and
# drops the other on the floor, which would silently uncap a local model. No
# server takes both, so: the old name first, and the new one after an endpoint
# has rejected it once.
_CAP_FIELD: dict[str, str] = {}

# Our effort levels onto the ones an OpenAI-compatible server accepts. `none`
# is passed through: OpenAI's chat completions endpoint refuses function tools
# on a reasoning model at any other setting.
_EFFORT = {
    "none": "none",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


# A rate limit is a queue, not a failure. Without this an optimiser run dies
# partway through and the tokens it already spent buy nothing: OpenRouter caps
# new accounts at 20 requests a minute per model, and one cold turn is five to
# eight calls, so three concurrent rollouts exceed it in seconds.
#
# Bounded, because a limit that never clears is a configuration problem and
# waiting quietly on one is worse than saying so.
_RATE_LIMIT_TRIES = 6
_RATE_LIMIT_CEILING = 60.0


async def _post(spec: Model, body: dict[str, Any]) -> Any:
    """One request, retried while the endpoint says it is rate limited.

    `Retry-After` when the endpoint sends one, and OpenRouter's reset timestamp
    when it sends that instead; otherwise exponential backoff. A 429 costs no
    tokens, so retrying is free apart from the wall clock.
    """
    client = _http_client(spec)
    for attempt in range(_RATE_LIMIT_TRIES):
        resp = await client.post("/chat/completions", json=body)
        if resp.status_code != 429 or attempt == _RATE_LIMIT_TRIES - 1:
            return resp
        delay = _retry_after(resp, attempt)
        log.warning(
            "rate limited by %s, waiting %.1fs (attempt %d of %d)",
            spec.url, delay, attempt + 1, _RATE_LIMIT_TRIES,
        )
        await asyncio.sleep(delay)
    return resp


def _retry_after(resp: Any, attempt: int) -> float:
    """How long the endpoint asked for, or a doubling backoff if it did not."""
    header = resp.headers.get("retry-after")
    if header:
        try:
            return min(float(header), _RATE_LIMIT_CEILING)
        except ValueError:
            pass

    # OpenRouter puts a millisecond epoch in the body rather than a header.
    try:
        headers = (resp.json().get("error") or {}).get("metadata", {}).get("headers", {})
        reset = float(headers.get("X-RateLimit-Reset", 0)) / 1000
    except Exception:
        reset = 0
    if reset:
        wait = reset - time.time()
        if 0 < wait <= _RATE_LIMIT_CEILING:
            return wait

    return min(2.0**attempt, _RATE_LIMIT_CEILING)


async def _complete_openai(
    *,
    spec: Model,
    system: str,
    messages: list[dict[str, Any]],
    effort: str,
    tools: list[dict[str, Any]] | None,
    schema: dict[str, Any] | None,
    max_tokens: int,
    cache_system: bool = False,
) -> Result:
    router = spec.provider == "openrouter"

    # The system block, plain or marked for caching. OpenRouter passes the
    # marker through to Anthropic, where it is what makes a cached turn cheap;
    # `graph.plan` is the node that asks for it. A plain string everywhere else,
    # because an endpoint that does not know the block form rejects the request
    # rather than ignoring the marker.
    system_content: Any = system
    if router and cache_system:
        system_content = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]

    body: dict[str, Any] = {
        "model": spec.model,
        "messages": [{"role": "system", "content": system_content}, *messages],
    }
    # Local models don't need Opus 5's headroom for adaptive thinking, and the
    # ceiling on output tokens is the ceiling on wall clock. On OpenRouter it is
    # also the thinking budget: effort is applied as a fraction of this, and
    # thinking bills as output. Lowering it is the most direct cost lever there.
    cap = min(max_tokens, spec.max_tokens)
    cap_field = _CAP_FIELD.get(spec.url or "", "max_tokens")
    body[cap_field] = cap

    if router:
        # An object, not the flat field, which OpenRouter does not read. All six
        # levels survive: it defines `max` and `xhigh`, so the two that the
        # OpenAI mapping below flattens into `high` stay distinct — which is the
        # difference between per-node effort being a real setting and a
        # decoration.
        body["reasoning"] = {"effort": effort}
        # What the call charged, on the response. Spend is otherwise a surprise
        # at the end of the month.
        body["usage"] = {"include": True}
    else:
        # Per-node effort applies here too — without it a reasoning model has no
        # brake, and `plan` is configured `low` precisely so a cached turn is cheap.
        body["reasoning_effort"] = _EFFORT.get(effort, "medium")

    if schema is not None:
        # Structured output as a forced tool call. `response_format` and the
        # native `format` parameter are both silently ignored on Ollama's MLX
        # runner, which has no grammar-constrained decoding; the tool channel is
        # JSON by construction.
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": _EMIT,
                    "description": "Return the answer in this exact shape.",
                    "parameters": schema,
                },
            }
        ]
        # The string, not OpenAI's named-function form — equivalent here, since
        # the tools list holds exactly one entry, and portable, which the named
        # form is not: LM Studio rejects it with "Invalid tool_choice type".
        body["tool_choice"] = "required"
    elif tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    resp = await _post(spec, body)
    if (
        resp.status_code == 400
        and cap_field == "max_tokens"
        and "max_completion_tokens" in resp.text
    ):
        # The endpoint wants the other spelling. Remember, and go again — a
        # rejected request costs no tokens, and this happens once per endpoint
        # per process.
        _CAP_FIELD[spec.url or ""] = "max_completion_tokens"
        del body["max_tokens"]
        body["max_completion_tokens"] = cap
        resp = await _post(spec, body)
    if resp.status_code >= 400:
        # Not `raise_for_status()`, which throws away the body — and the body is
        # the entire diagnosis when a local server rejects one field.
        raise LlmError(
            f"{resp.status_code} from {spec.url}"
            f"/chat/completions: {resp.text[:500]}"
        )
    data = resp.json()

    choice = data["choices"][0]
    message = choice["message"]
    usage = data.get("usage") or {}
    # OpenRouter reports both when asked; every other endpoint reports neither,
    # and `None` for the cost says "nobody said", which is not "free".
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    charged = usage.get("cost")
    cost = float(charged) if charged is not None else None
    # Reasoning models leak <think>…</think> into content; it is not an answer.
    text = _THINK.sub("", message.get("content") or "").strip()

    calls = [
        ToolUse(
            id=c["id"],
            name=c["function"]["name"],
            input=json.loads(c["function"]["arguments"] or "{}"),
        )
        for c in message.get("tool_calls") or []
    ]

    if schema is not None:
        emitted = next((c for c in calls if c.name == _EMIT), None)
        if emitted is None:
            # finish_reason is the diagnosis: "length" means max_tokens cut the
            # model off while it was still thinking, so it never reached the call.
            raise ValueError(
                f"no {_EMIT!r} tool call: finish_reason="
                f"{choice.get('finish_reason')!r}, "
                f"completion_tokens={usage.get('completion_tokens')}, "
                # `cap`, not `body["max_tokens"]`: the negotiation above may
                # have renamed that key, and a KeyError here would replace the
                # diagnosis with a worse one.
                f"max_tokens={cap}, text={text[:160]!r}"
            )
        # Hand it back as JSON text so .parsed() works identically on both backends.
        return Result(
            text=json.dumps(emitted.input),
            stop_reason="end_turn",
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            cache_read=cached,
            cost=cost,
        )

    return Result(
        text=text,
        tool_uses=calls,
        stop_reason=choice.get("finish_reason"),
        tokens_in=usage.get("prompt_tokens", 0),
        tokens_out=usage.get("completion_tokens", 0),
        cache_read=cached,
        cost=cost,
    )


# ---------------------------------------------------------------------- dispatch


async def complete(
    *,
    system: str,
    messages: list[dict[str, Any]],
    effort: str,
    tools: list[dict[str, Any]] | None = None,
    schema: dict[str, Any] | None = None,
    max_tokens: int = MAX_TOKENS,
    cache_system: bool = False,
    node: str = "model",
) -> Result:
    """One model call.

    `cache_system` reaches Anthropic directly and through OpenRouter, which
    passes the marker on; every other endpoint ignores it.

    `node` names the caller for the trace, selects the prompt and selects the
    model. This is also the only place a generation is opened, which is what
    makes each explore-loop iteration visible rather than summed inside the node.
    """
    assert not (tools and schema), "a structured call takes no tools"
    spec = config().model_for(node)
    anthropic_backend = spec.provider == "anthropic"

    with tracing.generation(
        name=node,
        model=spec.model,
        input={"system": system, "messages": messages},
        metadata={
            "provider": spec.provider,
            "effort": effort,
            "cache_system": cache_system,
            "structured": schema is not None,
            "tools": [t["name"] for t in tools or []],
            "max_tokens": max_tokens,
        },
    ) as gen:
        if anthropic_backend:
            result = await _complete_anthropic(
                spec=spec,
                system=system,
                messages=messages,
                effort=effort,
                tools=tools,
                schema=schema,
                max_tokens=max_tokens,
                cache_system=cache_system,
            )
        else:
            result = await _complete_openai(
                spec=spec,
                system=system,
                messages=messages,
                effort=effort,
                tools=tools,
                schema=schema,
                max_tokens=max_tokens,
                cache_system=cache_system,
            )

        gen.update(
            output={
                "text": result.text,
                "tool_uses": [{"name": t.name, "input": t.input} for t in result.tool_uses],
            },
            # `cache_read` is what catches a regression on the cached `plan`
            # path, which is what prompt caching is there for.
            usage_details={
                "input": result.tokens_in,
                "output": result.tokens_out,
                "cache_read_input_tokens": result.cache_read,
            },
            metadata={"stop_reason": result.stop_reason},
        )
        return result


async def aclose() -> None:
    for client in _http.values():
        await client.aclose()
    _http.clear()
