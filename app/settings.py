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

    # The address of the `default` registry row, the only one the environment
    # owns. Every other target comes out of the `connection` table.
    target_database_url: str = "postgresql://reader:reader@localhost:5432/business"

    # The owner connection. Nothing in app/ reads it — it is for the test
    # harness and for applying demo/demo.sql from outside Docker.
    target_admin_url: str = "postgresql://business:business@localhost:5432/business"

    # --- the API ------------------------------------------------------------
    # Everything under /v1 requires this token. Empty disables enforcement and
    # the server warns at startup.
    api_token: str = ""

    # A urlsafe-base64 32-byte Fernet key sealing registered warehouse
    # passwords. Empty means plaintext at rest, and a startup warning.
    connection_secret: str = ""

    # --- model credentials --------------------------------------------------
    # Which model, at what effort, is config/config.yaml. Only the keys are here.
    anthropic_api_key: str = ""  # falls back to the SDK's own env resolution
    openai_api_key: str = "not-needed"

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
