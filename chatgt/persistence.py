#!/usr/bin/env python3

import json
import os
from typing import Any, Dict, List

import app_state as state


PERSIST_PROGRESS = os.getenv("AGENT_PERSIST_PROGRESS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PERSIST_STATUS = os.getenv("AGENT_PERSIST_STATUS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def should_persist_message(msg: Dict[str, Any]) -> bool:
    role = str(msg.get("role", ""))

    if role in {"user", "agent"}:
        return True

    if role == "progress":
        return PERSIST_PROGRESS

    if role == "status":
        return PERSIST_STATUS

    return False


def persisted_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [msg for msg in messages if should_persist_message(msg)]


def load_settings() -> None:
    try:
        if os.path.exists(state.SETTINGS_FILE):
            with open(state.SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            selected_model = data.get("model")
            if isinstance(selected_model, str) and selected_model.strip():
                state.agent.model = selected_model.strip()
                print(f"Loaded selected model: {state.agent.model}")

            state.agent.set_think_mode(data.get("think_mode"))
            print(f"Loaded think mode: {state.agent.get_think_mode_label()}")

            context_settings = data.get("context")
            if isinstance(context_settings, dict):
                state.update_context_settings(context_settings)
                print(
                    "Loaded context settings: "
                    f"{state.RECENT_CONVERSATION_CONTEXT_MESSAGES} messages, "
                    f"{state.RECENT_CONVERSATION_CONTEXT_CHARS} chars/message, "
                    f"{state.CONTEXT_TARGET_TOKENS} target tokens"
                )

            routing_settings = data.get("routing")
            if isinstance(routing_settings, dict):
                state.update_routing_settings(routing_settings)
                print(
                    "Loaded routing settings: "
                    f"enabled={state.agent.routing_enabled}, "
                    f"quality={state.agent.routing_quality_mode}, "
                    f"debug={state.agent.routing_debug_enabled}"
                )

    except Exception as exc:
        print(f"Could not load settings file: {exc}")


def save_settings() -> None:
    try:
        os.makedirs(os.path.dirname(state.SETTINGS_FILE), exist_ok=True)

        with open(state.SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": state.agent.model,
                    "think_mode": state.agent.think_mode,
                    "context": {
                        "recent_context_messages": state.RECENT_CONVERSATION_CONTEXT_MESSAGES,
                        "recent_context_chars": state.RECENT_CONVERSATION_CONTEXT_CHARS,
                        "context_target_tokens": state.CONTEXT_TARGET_TOKENS,
                        "ollama_num_ctx": state.agent.ollama_num_ctx,
                    },
                    "routing": {
                        "enabled": state.agent.routing_enabled,
                        "quality_mode": state.agent.routing_quality_mode,
                        "debug_logging": state.agent.routing_debug_enabled,
                        "roles": state.agent.routing_roles,
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    except Exception as exc:
        print(f"Could not save settings file: {exc}")


def load_messages() -> None:
    try:
        if os.path.exists(state.SESSION_FILE):
            with open(state.SESSION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                compacted_data = persisted_messages(data)
                state.messages.clear()
                state.messages.extend(compacted_data)
                print(
                    f"Loaded {len(state.messages)} saved messages "
                    f"from {len(data)} stored records."
                )

                if len(compacted_data) != len(data):
                    save_messages()
    except Exception as exc:
        print(f"Could not load session file: {exc}")


def save_messages() -> None:
    try:
        os.makedirs(os.path.dirname(state.SESSION_FILE), exist_ok=True)

        with open(state.SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(persisted_messages(state.messages), f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"Could not save session file: {exc}")


def load_memory_state() -> Dict[str, Any]:
    try:
        if os.path.exists(state.MEMORY_STATE_FILE):
            with open(state.MEMORY_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                return data
    except Exception as exc:
        print(f"Could not load memory state file: {exc}")

    return {
        "summarized_until_index": 0,
    }


def save_memory_state(memory_state: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(state.MEMORY_STATE_FILE), exist_ok=True)

        with open(state.MEMORY_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(memory_state, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"Could not save memory state file: {exc}")
