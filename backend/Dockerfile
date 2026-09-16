# syntax=docker/dockerfile:1.7
#
# The cloud processes: the Web API and the Embed Worker, built from one base.
#
#   docker build --target api    -t interroai-api    backend
#   docker build --target worker -t interroai-worker backend
#
# The images install requirements-cloud.lock and nothing else: fully pinned,
# hash-checked, and resolved for every platform. Neither carries the desktop
# runtime's dependencies (ChromaDB, the chunker, Redis), and
# tests/test_dependency_split.py keeps the code honest about that.
#
# Migrations run from the API image as a one-off, with
# INTERROAI_MIGRATIONS_DATABASE_URL set:
#   docker run --rm -e INTERROAI_MIGRATIONS_DATABASE_URL=... interroai-api \
#     alembic -c cloud/alembic.ini upgrade head

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    INTERROAI_LOG_FORMAT=json

COPY --from=ghcr.io/astral-sh/uv:0.11.12 /uv /usr/local/bin/uv

WORKDIR /app

COPY requirements-cloud.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip sync --system --require-hashes requirements-cloud.lock

# Only what the cloud processes import: no agents, no runtime API, no tests.
COPY contracts ./contracts
COPY core ./core
COPY cloud ./cloud

RUN useradd --system --no-create-home --uid 10001 interroai
USER interroai


FROM base AS api

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"]

# --proxy-headers: behind the Container Apps ingress, its only way in, the client
# address comes from X-Forwarded-For, and the sign-in rate limits need the real
# one. The access log is the API's own (cloud/api/middleware.py), so uvicorn's
# is off.
CMD ["uvicorn", "cloud.api.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--no-access-log"]


FROM base AS worker

CMD ["python", "-m", "cloud.worker"]
