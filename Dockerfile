# Ares A2A agent servers (orchestrator, coder, thinker, runner, serena).
# Ollama stays on the host; see docker/agents/docker-compose.yml.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_TOOL_DIR=/opt/uv-tools \
    UV_TOOL_BIN_DIR=/usr/local/bin

# Serena for the serena agent. Its Python language server (pyright) needs
# Node; pyright[nodejs] bundles both, so /serena needs no network at runtime.
RUN uv tool install --no-cache serena-agent==1.3.0 --with "pyright[nodejs]" \
    && /opt/uv-tools/serena-agent/bin/pyright --version

WORKDIR /app
# Dependencies first so source edits don't reinstall them.
COPY pyproject.toml README.md LICENSE ./
RUN uv pip install --system --no-cache -r pyproject.toml
COPY ares ./ares
RUN uv pip install --system --no-cache --no-deps .

# Runs as the host user (compose sets it), so these must be writable by anyone.
RUN mkdir -p /home/ares /data /workspace && chmod 1777 /home/ares /data
ENV HOME=/home/ares \
    ARES_TRACE_DB_PATH=/data/traces.db \
    ARES_FEEDBACK_PATH=/data/feedback.jsonl \
    ARES_SERENA_PROJECT=/workspace

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; [urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=3) for p in range(8100, 8105)]"

CMD ["python", "-m", "ares.a2a_server"]
