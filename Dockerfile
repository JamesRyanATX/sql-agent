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
# anything. `package = false` in pyproject.toml means there is no project to
# install — just the locked dependency set.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app

EXPOSE 8000

# --reload is the dev loop: app/ is bind-mounted, so an edit on the host
# restarts the server in place with no rebuild.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--reload", "--reload-dir", "/srv/app"]
