FROM ubuntu:24.04

# Install uv (which manages Python versions automatically)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root user for execution
RUN useradd --create-home --shell /bin/bash executor

# Set up cache directory (owned by executor so named-volume mounts inherit perms)
RUN mkdir -p /cache && chown executor:executor /cache
VOLUME /cache
ENV UV_CACHE_DIR=/cache

USER executor

WORKDIR /work
