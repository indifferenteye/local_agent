``` bash

starting docker? serve?

==

docker build -t ollama-agent .

==

docker run -it --rm `
  -p 8080:8080 `
  -v "${PWD}\chatgt:/app:ro" `
  -v "${PWD}\workdir:/agent/workdir" `
  -w /app `
  -e OLLAMA_URL="http://host.docker.internal:11434" `
  -e OLLAMA_MODEL="gemma4:e2b" `
  -e AGENT_WORKDIR="/agent/workdir" `
  -e AGENT_MAX_ITERATIONS="24" `
  -e AGENT_SUMMARIZE_AFTER_MESSAGES="60" `
  -e AGENT_RECENT_MESSAGES_TO_KEEP="30" `
  -e AGENT_RECENT_CONVERSATION_CONTEXT_MESSAGES="20" `
  -e OLLAMA_NUM_PREDICT="4096" `
  ollama-agent

==

  create a html file with a canvas that animates a rotating donut. make it pretty

==

```

= access via other device
ipconfig
http://192.168.178.42:8080

TODO: 
- reasoning?
- queue tasks
- skills
- better system promt, better tools
- multiple model workflow? (agentic, coding, images etc)
- ... 
