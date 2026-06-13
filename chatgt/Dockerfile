FROM python:3.11-slim

ENV OLLAMA_URL=http://localhost:11434 \
    OLLAMA_MODEL=gemma4:e2b \
    AGENT_WORKDIR=/agent/workdir

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    zstd \
    bash \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /agent
RUN mkdir -p /agent/workdir

COPY agent-core-fixed.py /agent/agent-core.py

RUN pip install --no-cache-dir requests

EXPOSE 11434

RUN cat > /agent/start.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

ollama serve &
OLLAMA_PID=$!

for i in $(seq 1 60); do
  if curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

ollama pull "$OLLAMA_MODEL"
python3 /agent/agent-core.py "$@"

wait "$OLLAMA_PID"
SH
RUN chmod +x /agent/start.sh

CMD ["/agent/start.sh"]
