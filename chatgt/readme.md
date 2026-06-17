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
  -e AGENT_RECENT_CONVERSATION_CONTEXT_CHARS="6000" `
  -e AGENT_CONTEXT_TARGET_TOKENS="16000" `
  -e OLLAMA_NUM_CTX="64000" `
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

Context notes:
- The app keeps recent user/agent messages verbatim and compresses older messages into `.agent_memory_summary.txt`.
- `AGENT_CONTEXT_TARGET_TOKENS` is the app's prompt budgeting target.
- `OLLAMA_NUM_CTX` is sent to Ollama as `options.num_ctx` when set.
- For larger local-agent workloads, also configure the Ollama server/app context length. Example: `OLLAMA_CONTEXT_LENGTH=64000 ollama serve`.
- The app does not persist Ollama's deprecated generated token `context`, so switching models between messages keeps working from text history and summaries.

Session persistence notes:
- `.agent_sessions.json` persists durable chat messages by default: `user` and final `agent` messages.
- Live `progress` and `status` messages still appear in the UI, but are not saved by default.
- Set `AGENT_PERSIST_PROGRESS=true` or `AGENT_PERSIST_STATUS=true` only when debugging detailed runtime traces.
