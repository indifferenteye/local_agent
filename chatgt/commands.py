#!/usr/bin/env python3

import app_state as state
from events import broadcast
from persistence import save_settings


def slash_command_help_text() -> str:
    return """Available commands:

/help
Show available commands.

/set think
Enable model thinking if the selected model supports it.

/set nothink
Disable model thinking if the selected model supports it.

/set think low
Use low thinking effort if the selected model supports it.

/set think medium
Use medium thinking effort if the selected model supports it.

/set think high
Use high thinking effort if the selected model supports it.

/set think default
Use Ollama/model default behavior.
"""


def handle_slash_command(task: str) -> bool:
    command = task.strip()
    command_lower = command.lower()

    if not command_lower.startswith("/"):
        return False

    broadcast("user", command)

    if command_lower in {"/help", "/commands"}:
        broadcast("agent", slash_command_help_text())
        return True

    if command_lower == "/set nothink":
        state.agent.set_think_mode(False)
        save_settings()
        broadcast("agent", "Thinking disabled. Current mode: nothink.")
        broadcast("status", "settings changed")
        return True

    if command_lower == "/set think":
        state.agent.set_think_mode(True)
        save_settings()
        broadcast("agent", "Thinking enabled. Current mode: think.")
        broadcast("status", "settings changed")
        return True

    if command_lower.startswith("/set think "):
        value = command_lower.replace("/set think ", "", 1).strip()

        if value in {"low", "medium", "high"}:
            state.agent.set_think_mode(value)
            save_settings()
            broadcast("agent", f"Thinking mode set to {value}.")
            broadcast("status", "settings changed")
            return True

        if value in {"default", "none", "null"}:
            state.agent.set_think_mode(None)
            save_settings()
            broadcast("agent", "Thinking mode reset to Ollama/model default.")
            broadcast("status", "settings changed")
            return True

        broadcast(
            "agent",
            "Unknown thinking mode. Use: /set think, /set nothink, /set think low, /set think medium, /set think high, or /set think default.",
        )
        return True

    broadcast("agent", "Unknown command. Type /help to see available commands.")
    return True
