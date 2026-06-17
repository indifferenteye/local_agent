#!/usr/bin/env python3

import os
import queue
import threading
from typing import Any, Dict, List

from agent_core import OllamaAgent
from context_manager import ContextManager, clamp_int


agent = OllamaAgent()
agent_lock = threading.Lock()

SESSION_FILE = os.path.join(agent.working_dir, ".agent_sessions.json")
SETTINGS_FILE = os.path.join(agent.working_dir, ".agent_settings.json")
MEMORY_STATE_FILE = os.path.join(agent.working_dir, ".agent_memory_state.json")

RECENT_MESSAGES_TO_KEEP = int(os.getenv("AGENT_RECENT_MESSAGES_TO_KEEP", "30"))
SUMMARIZE_AFTER_MESSAGES = int(os.getenv("AGENT_SUMMARIZE_AFTER_MESSAGES", "60"))
RECENT_CONVERSATION_CONTEXT_MESSAGES = int(
    os.getenv("AGENT_RECENT_CONVERSATION_CONTEXT_MESSAGES", "20")
)
RECENT_CONVERSATION_CONTEXT_CHARS = int(
    os.getenv("AGENT_RECENT_CONVERSATION_CONTEXT_CHARS", "6000")
)
CONTEXT_TARGET_TOKENS = int(os.getenv("AGENT_CONTEXT_TARGET_TOKENS", "16000"))

context_manager = ContextManager(
    recent_message_limit=RECENT_CONVERSATION_CONTEXT_MESSAGES,
    recent_message_char_limit=RECENT_CONVERSATION_CONTEXT_CHARS,
    target_tokens=CONTEXT_TARGET_TOKENS,
)


def update_context_settings(data: Dict[str, Any]) -> None:
    global RECENT_CONVERSATION_CONTEXT_MESSAGES
    global RECENT_CONVERSATION_CONTEXT_CHARS
    global CONTEXT_TARGET_TOKENS

    RECENT_CONVERSATION_CONTEXT_MESSAGES = clamp_int(
        data.get("recent_context_messages"),
        RECENT_CONVERSATION_CONTEXT_MESSAGES,
        2,
        100,
    )
    RECENT_CONVERSATION_CONTEXT_CHARS = clamp_int(
        data.get("recent_context_chars"),
        RECENT_CONVERSATION_CONTEXT_CHARS,
        500,
        50000,
    )
    CONTEXT_TARGET_TOKENS = clamp_int(
        data.get("context_target_tokens"),
        CONTEXT_TARGET_TOKENS,
        2048,
        256000,
    )
    ollama_num_ctx = data.get("ollama_num_ctx")
    if str(ollama_num_ctx or "").strip():
        agent.ollama_num_ctx = agent.parse_optional_int(ollama_num_ctx)
    elif "ollama_num_ctx" in data:
        agent.ollama_num_ctx = None

    context_manager.recent_message_limit = RECENT_CONVERSATION_CONTEXT_MESSAGES
    context_manager.recent_message_char_limit = RECENT_CONVERSATION_CONTEXT_CHARS
    context_manager.target_tokens = CONTEXT_TARGET_TOKENS

messages: List[Dict[str, Any]] = []
subscribers: List[queue.Queue] = []
running = False
