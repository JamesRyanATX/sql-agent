from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- two databases, deliberately ----------------------------------------
    # No single DATABASE_URL and no fallback between these: a fallback is how
    # both connections end up back on one server.
    agent_database_url: str = "postgresql://agent:agent@localhost:5433/agent"

    # The database the agent answers questions about. Postgres, MySQL or SQLite;
    # a bare scheme is mapped onto an installed driver by `db.target_url`.
    target_database_url: str = "postgresql://reader:reader@localhost:5432/business"

    # The suite's superuser handle, and nothing in app/ reads it. It creates and
    # drops the test databases, loads their data as owner, and is the other half
    # of the isolation tests that prove `reader` cannot write. Named TEST_ rather
    # than TARGET_ because it is not a target: the agent never connects as this
    # role and its engines cannot issue DDL at all.
    test_admin_url: str = "postgresql://business:business@localhost:5432/business"

    # --- the API ------------------------------------------------------------
    # Everything under /v1 requires this token. Empty disables enforcement and
    # the server warns at startup.
    api_token: str = ""

    # --- model credentials --------------------------------------------------
    # Which model, at what effort, is config/config.yaml. Only the keys are here.
    anthropic_api_key: str = ""  # falls back to the SDK's own env resolution
    openai_api_key: str = "not-needed"
    # OpenRouter bills separately from OpenAI, so it gets its own name rather
    # than overwriting a key that is still wanted for something else. Falls back
    # to `openai_api_key` when unset, because one gateway at a time is the
    # common case and two names for one value is a trap.
    openrouter_api_key: str = ""

    # Holds config.yaml, config.local.yaml and prompts/. An environment variable
    # because where the config lives cannot itself be config.
    config_dir: str = "config"

    # --- observability ------------------------------------------------------
    # Tracing is on when both keys are set and off otherwise; there is no third
    # `enabled` flag to disagree with them. On captures prompts, SQL and rows.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    # docker-compose.yml overrides this for the API container.
    langfuse_host: str = "http://localhost:3000"

    # Server-side refusal fallbacks. Turn off if the account lacks the beta.
    use_fallbacks: bool = True

    # Per-target pools: a registry of ten should not mean ten idle sockets.
    target_pool_max: int = 5
    target_pool_max_idle: float = 300.0
    target_connect_timeout: float = 10.0


@lru_cache
def settings() -> Settings:
    return Settings()
