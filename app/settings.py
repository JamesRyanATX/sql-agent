from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- two databases, deliberately -------------------------------------
    # The agent's memory and the data it queries live on separate servers, so
    # neither can reach the other. There is no single DATABASE_URL and no
    # fallback between these: a fallback is exactly how both connections end
    # up back on one server, which is the thing being prevented.
    agent_database_url: str = "postgresql://agent:agent@localhost:5433/agent"

    # The agent connects as `reader`, which holds SELECT and nothing else.
    # Named for the role it plays, not for the demo fixture that fills it —
    # point this at a real warehouse and the name still reads true.
    #
    # This is now one connection among however many are registered: it is the
    # address of the `default` row, the only one the environment owns. Every
    # other target comes out of the `connection` table.
    target_database_url: str = "postgresql://reader:reader@localhost:5432/business"

    # The owner connection. Nothing in app/ reads it: it exists for the test
    # harness, which has to create and populate a database, and for applying
    # demo/demo.sql from outside Docker.
    target_admin_url: str = "postgresql://business:business@localhost:5432/business"

    # --- the API -----------------------------------------------------------
    # Where the client looks for the server is the client's business: the CLI
    # reads SQL_AGENT_URL itself and imports nothing from here.
    # Everything under /v1 requires this token. Empty disables enforcement,
    # which is what tests and a first `make up` run on; the server says so
    # at startup rather than leaving it silent.
    api_token: str = ""

    # A urlsafe-base64 32-byte Fernet key. Set, and registered warehouse
    # passwords are encrypted at rest on agent-db; empty, and they are stored in
    # plaintext and the server warns at startup — the same bargain api_token
    # makes above. See app/secrets.py for what this does and does not buy.
    connection_secret: str = ""

    # "anthropic" is the demo model. "openai_compat" points at any OpenAI-shaped
    # endpoint (Ollama, vLLM, LM Studio) for local development.
    provider: str = "anthropic"

    anthropic_api_key: str = ""  # falls back to the SDK's own env resolution
    model: str = "claude-opus-5"

    openai_base_url: str = "http://localhost:11434/v1"
    openai_model: str = ""
    openai_api_key: str = "not-needed"
    openai_timeout: float = 900.0  # a 27B local model takes minutes, not seconds

    # Generation controls for local reasoning models. A thinking model handed an
    # open-ended question will happily spend ten minutes deliberating before its
    # first tool call, so the ceiling on output tokens is the ceiling on wall
    # clock. Caps the per-call max_tokens the nodes ask for.
    openai_max_tokens: int = 16_000
    # Sent only if set — support varies by server and model.
    openai_reasoning_effort: str = ""

    # Per-node effort. See PLAN.md §7.1: lowering effort is how we make a node
    # cheap. Disabling thinking is not — on Opus 5 that can turn a tool call
    # into plain visible text that never runs, which would silently break the
    # explore loop.
    effort_plan: str = "low"
    effort_explore: str = "high"
    effort_sql: str = "high"
    effort_extract: str = "low"

    # --- observability -----------------------------------------------------
    # Tracing is on when both keys are set and off otherwise. There is
    # deliberately no third `enabled` flag: a flag is a state that can disagree
    # with the keys, and "on but unconfigured" is a startup warning nobody reads.
    #
    # On means the prompts, the generated SQL and the result rows go into the
    # trace — that is what makes a trace worth opening, and there is no useful
    # half-measure. The store is the container stack behind
    # `make langfuse-up`, so nothing leaves the machine; it still holds whatever
    # the registered warehouse holds. See app/tracing.py.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    # localhost for host-side use; docker-compose.yml overrides it for the API
    # container, where localhost is the container.
    langfuse_host: str = "http://localhost:3000"

    # Server-side refusal fallbacks. Low-probability for schema introspection,
    # but a refusal mid-demo is unrecoverable, and the cost of having it on is
    # zero when it never fires. Turn off if the account lacks the beta.
    use_fallbacks: bool = True

    # Bounds, so T1 doesn't run forever on stage.
    max_tool_calls: int = 24
    max_fix_attempts: int = 3
    # Milliseconds, an integer, because that is the only unit all three
    # supported dialects can be told. It was the string "5s" — a Postgres
    # interval literal, which MySQL parses as neither a number nor an error.
    # See app/dialects.py for what each one does with it.
    statement_timeout_ms: int = 5_000
    max_rows: int = 50  # rows handed back to the model from execute

    # Per-target pools. Small and short-lived by default: these are somebody
    # else's databases, and a registry of ten should not mean ten idle sockets.
    target_pool_max: int = 5
    target_pool_max_idle: float = 300.0
    # A registered warehouse that is down should say so in seconds, not hang the
    # turn that asked about it.
    target_connect_timeout: float = 10.0


@lru_cache
def settings() -> Settings:
    return Settings()
