``` bash

starting docker? serve?

==

docker build -t ollama-agent .

==

docker run -it --rm `
  -p 8080:8080 `
  -v "${PWD}\chatgt\web-agent.py:/agent/web-agent.py:ro" `
  -v "${PWD}\chatgt\agent_core_fixed_import.py:/agent/agent_core_fixed_import.py:ro" `
  -v "${PWD}\workdir:/agent/workdir" `
  -w /agent `
  -e OLLAMA_URL="http://host.docker.internal:11434" `
  -e OLLAMA_MODEL="gemma4:e2b" `
  -e AGENT_WORKDIR="/agent/workdir" `
  -e AGENT_MAX_ITERATIONS="20" `
  ollama-agent

==

  create a html file with a canvas that animates a rotating donut. make it pretty

==

```

= access via other device
ipconfig
http://192.168.178.42:8080

TODO: 
- multiple steps?
- better ui? Webui?
- reasoning?
- testing
- better feedback after task comletion (and during?)
- queue tasks
- skills
- ... 
