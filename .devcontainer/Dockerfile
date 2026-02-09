FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    bubblewrap ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
