# The API. Postgres comes from the `db` service; see docker-compose.yml.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# The source tree is bind-mounted over /srv at run time, so the virtualenv has
# to live somewhere the mount can't shadow — otherwise the container starts
# with no dependencies installed and nothing says why.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /srv

# Dependencies only, and before the source, so editing a node doesn't reinstall
# anything. --no-install-project skips building the project's own wheel, which
# is the CLI: this image runs the server and never invokes it, and the wheel
# could not build here anyway because cli/sql_agent/ is not copied in.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
# The prompts and config/config.yaml. Not optional and not mounted-only: an
# image that can start without them would start with no prompts at all, and the
# first symptom would be a turn rather than a boot.
COPY config ./config

EXPOSE 8000

# --reload is the dev loop: app/ and config/ are bind-mounted, so an edit on the
# host restarts the server in place with no rebuild. config/ is in the list
# because prompts and config both resolve once per process — without a restart,
# editing a prompt does nothing and says nothing.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--reload", "--reload-dir", "/srv/app", "--reload-dir", "/srv/config"]
