from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = "postgresql://flyline:flyline@localhost:5432/flyline"

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

    # Server-side refusal fallbacks. Low-probability for schema introspection,
    # but a refusal mid-demo is unrecoverable, and the cost of having it on is
    # zero when it never fires. Turn off if the account lacks the beta.
    use_fallbacks: bool = True

    # Bounds, so T1 doesn't run forever on stage.
    max_tool_calls: int = 24
    max_fix_attempts: int = 3
    statement_timeout: str = "5s"
    max_rows: int = 50  # rows handed back to the model from execute


@lru_cache
def settings() -> Settings:
    return Settings()
