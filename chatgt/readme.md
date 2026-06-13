``` bash

starting docker? serve?

==

docker run -it --rm `
  -v "${PWD}\chatgt\agent-core-fixed.py:/agent/agent-core-fixed.py:ro" `
  -v "${PWD}\workdir:/agent/workdir" `
  -w /agent `
  -e OLLAMA_URL="http://host.docker.internal:11434" `
  -e OLLAMA_MODEL="gemma4:e2b" `
  -e AGENT_WORKDIR="/agent/workdir" `
  ollama-agent `
  python /agent/agent-core-fixed.py

==

  create a html file with a canvas that animates a rotating donut. make it pretty