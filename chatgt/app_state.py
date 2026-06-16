#!/usr/bin/env python3

import os
import queue
import threading
from typing import Any, Dict, List

from agent_core import OllamaAgent


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

messages: List[Dict[str, Any]] = []
subscribers: List[queue.Queue] = []
running = False
