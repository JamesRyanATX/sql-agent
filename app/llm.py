"""The one place that talks to a model (PLAN.md §7.1).

Two backends behind one `complete()`:

- **anthropic** — `claude-opus-5`, the demo model. Adaptive thinking, effort,
  native structured outputs.
- **openai_compat** — any OpenAI-shaped endpoint, including Ollama. For local
  development when the Anthropic key isn't available.

Which of the two a call uses is `config/config.yaml`, resolved per `node=`: a
global `model:` block, and an optional override on any node. That is why
`assistant_turn()` and `tool_results()` take a `node` — they pick a wire format,
and the wire format is a property of the node's backend, not of the process.
Reading the global provider there was correct only while there was one.

Nodes never see the difference. Messages are built with `assistant_turn()` and
`tool_results()` rather than literal dicts, because the two wire formats
disagree about how a tool result is represented.

Structured output differs by backend and the difference is not cosmetic:
Anthropic constrains the response directly, while on an OpenAI-compatible
endpoint we ask for a single forced tool call whose parameters *are* the schema.
The tool channel is already JSON-shaped, so it works even where grammar-
constrained decoding isn't available — which is the case on Ollama's MLX runner,
where both `response_format` and the native `format` parameter are ignored.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx

from app import tracing
from app.config import Model, config
from app.settings import settings

# Generous, and deliberately so: on Opus 5 max_tokens caps thinking *plus*
# response text together, and thinking is on by default. Sized tight, answers
# truncate mid-sentence. 16k also keeps non-streaming requests inside the SDK's
# HTTP timeout.
MAX_TOKENS = 16_000

# The name of the synthetic tool used to carry structured output on
# openai_compat. Never collides with a real tool — schema calls pass no tools.
_EMIT = "respond"

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

_anthropic_client: anthropic.AsyncAnthropic | None = None
# Keyed by (url, timeout), because a per-node override may name another endpoint
# and httpx binds base_url and timeout to the client. Two nodes on one endpoint
# still share one connection pool, which is the reason not to build these per call.
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

    `stream_turn` renders this as one fatal error event, so whatever is in the
    message is what a user sees — which is why the server's own body goes in it.
    """


@dataclass(slots=True)
class Result:
    text: str = ""
    tool_uses: list[ToolUse] = field(default_factory=list)
    stop_reason: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read: int = 0
    raw: Any = None  # provider-native assistant content, for echo-back

    def parsed(self) -> dict[str, Any]:
        """The JSON a structured-output call produced."""
        return json.loads(self.text)


# ------------------------------------------------------------------- message API


def assistant_turn(result: Result, *, node: str) -> dict[str, Any]:
    """The assistant turn to append before handing back tool results.

    `node` for the same reason `complete()` takes one: it decides the backend,
    and the two backends disagree about the shape of an assistant turn. Reading
    a global provider here would build Anthropic's shape for a node overridden
    to openai_compat, and the explore loop is the only caller — so the failure
    would be the loop, every time, on the second iteration.
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
    """Make an assistant turn safe to echo back after a mid-output fallback.

    Thinking and tool_use blocks from before the boundary belong to a model that
    is no longer answering, and replaying them is rejected. No fallback block
    means nothing to do, which is the overwhelmingly common case.
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

    # The cached path re-sends the same system prompt and the whole cache on
    # every turn, which is exactly the shape prompt caching is for. Breakpoint
    # goes on the system block: it is the stable prefix, and the question after
    # it is the only part that varies.
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

    # Thinking is left unset — on Opus 5 that runs adaptive, which is what we
    # want. Cost is controlled with effort, never by disabling thinking: with
    # thinking off the model can write a tool call into its visible text instead
    # of emitting a tool_use block, and in the explore loop it would never run.
    if settings().use_fallbacks:
        resp = await _anthropic().beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
        )
    else:
        resp = await _anthropic().messages.create(**kwargs)

    # Check this before touching content: a refusal is a successful HTTP
    # response whose content is empty or partial.
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
        client = httpx.AsyncClient(
            base_url=spec.url or "",
            # A 27B model on consumer hardware takes minutes per call, not
            # seconds. The default 5s read timeout would fail every request.
            timeout=httpx.Timeout(spec.timeout, connect=10.0),
            headers={"authorization": f"Bearer {settings().openai_api_key}"},
        )
        _http[key] = client
    return client


# Our five effort levels onto the three an OpenAI-compatible server accepts.
_EFFORT = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


async def _complete_openai(
    *,
    spec: Model,
    system: str,
    messages: list[dict[str, Any]],
    effort: str,
    tools: list[dict[str, Any]] | None,
    schema: dict[str, Any] | None,
    max_tokens: int,
) -> Result:
    body: dict[str, Any] = {
        "model": spec.model,
        # Local models don't need Opus 5's headroom for adaptive thinking, and
        # the ceiling on output tokens is the ceiling on wall clock.
        "max_tokens": min(max_tokens, spec.max_tokens),
        "messages": [{"role": "system", "content": system}, *messages],
    }
    # Per-node effort applies here too. Without it a reasoning model has no
    # brake: the `plan` node is configured `low` precisely because a cached
    # turn should be cheap, and a live run that ignored it spent 14,339 output
    # tokens deliberating — making the cached path *more* expensive than the
    # exploration it replaced.
    #
    # There used to be an OPENAI_REASONING_EFFORT that overrode every node at
    # once. It is gone: the per-node values in config/config.yaml are what it
    # was reaching for, and a global override of a per-node setting can only
    # ever be a way to lose the distinction.
    body["reasoning_effort"] = _EFFORT.get(effort, "medium")

    if schema is not None:
        # Structured output as a forced tool call. `response_format` and the
        # native `format` parameter are both silently ignored on Ollama's MLX
        # runner — they need grammar-constrained decoding, which MLX doesn't
        # implement — but the tool channel is JSON by construction.
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
        # `"required"` — the string — rather than OpenAI's named-function form
        # `{"type": "function", "function": {"name": ...}}`. The two mean the
        # same thing here, because the tools list above holds exactly one entry:
        # "call some tool" and "call `emit`" are the same instruction. The named
        # form is not universally implemented — LM Studio answers it with
        #   400 {"error":"Invalid tool_choice type: 'object'.
        #        Supported string values: none, auto, required"}
        # — while `"required"` is accepted by every OpenAI-shaped server tried,
        # including OpenAI itself. Prefer the spelling that is equivalent and
        # portable over the one that is more specific and not.
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

    resp = await _http_client(spec).post("/chat/completions", json=body)
    if resp.status_code >= 400:
        # Not `raise_for_status()`. It reports the status and the URL and throws
        # the body away — and the body is the entire diagnosis. A local server
        # rejecting one field answers "400 Bad Request for url ..." plus a link
        # to MDN, which sends you looking at the network instead of at the one
        # key it did not like.
        raise LlmError(
            f"{resp.status_code} from {spec.url}"
            f"/chat/completions: {resp.text[:500]}"
        )
    data = resp.json()

    choice = data["choices"][0]
    message = choice["message"]
    usage = data.get("usage") or {}
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
            # finish_reason is the whole diagnosis here: "length" means the
            # model was still thinking when max_tokens cut it off, so it never
            # reached the tool call. Without it this is an opaque empty string.
            raise ValueError(
                f"no {_EMIT!r} tool call: finish_reason="
                f"{choice.get('finish_reason')!r}, "
                f"completion_tokens={usage.get('completion_tokens')}, "
                f"max_tokens={body['max_tokens']}, text={text[:160]!r}"
            )
        # Hand it back as JSON text so .parsed() works identically on both backends.
        return Result(
            text=json.dumps(emitted.input),
            stop_reason="end_turn",
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
        )

    return Result(
        text=text,
        tool_uses=calls,
        stop_reason=choice.get("finish_reason"),
        tokens_in=usage.get("prompt_tokens", 0),
        tokens_out=usage.get("completion_tokens", 0),
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

    `effort` and `cache_system` are Anthropic-only and ignored elsewhere — an
    OpenAI-compatible endpoint has no equivalent of either.

    `node` names the caller for the trace, selects the prompt in `app/prompts.py`
    and now selects the model too — it is the one string that ties a graph node
    to everything configurable about it. Being the one place that talks to a
    model makes this also the one place a generation is opened: both backends,
    and every iteration of the explore loop, whose per-call cost is otherwise
    summed away inside the node before anything can see it. The wrapper goes here
    rather than in the two `_complete_*` functions so there is one thing to keep
    in step, not two.
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
            )

        gen.update(
            output={
                "text": result.text,
                "tool_uses": [{"name": t.name, "input": t.input} for t in result.tool_uses],
            },
            # `cache_read` has been on `Result` all along and read by nothing.
            # The cached `plan` path is exactly what prompt caching is for, so a
            # regression there is the one this number would catch.
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
