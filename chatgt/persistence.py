#!/usr/bin/env python3

import json
import os
from typing import Any, Dict

import app_state as state


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
                state.messages.clear()
                state.messages.extend(data)
                print(f"Loaded {len(state.messages)} saved messages.")
    except Exception as exc:
        print(f"Could not load session file: {exc}")


def save_messages() -> None:
    try:
        os.makedirs(os.path.dirname(state.SESSION_FILE), exist_ok=True)

        with open(state.SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(state.messages, f, indent=2, ensure_ascii=False)
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
