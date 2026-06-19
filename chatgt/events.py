#!/usr/bin/env python3

from datetime import datetime
from typing import Any, Dict

import app_state as state
from persistence import save_messages, should_persist_message


def normalize_progress_event(event: Any) -> Dict[str, Any]:
    if isinstance(event, dict):
        return {
            "role": "progress",
            "kind": str(event.get("kind", "progress")),
            "iteration": event.get("iteration"),
            "action": event.get("action"),
            "workflow_step": event.get("workflow_step"),
            "workflow_step_label": event.get("workflow_step_label"),
            "workflow_loop_iteration": event.get("workflow_loop_iteration"),
            "text": str(event.get("text", "")),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    return {
        "role": "progress",
        "kind": "progress",
        "iteration": None,
        "action": None,
        "workflow_step": None,
        "workflow_step_label": None,
        "workflow_loop_iteration": None,
        "text": str(event),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def broadcast_message(msg: Dict[str, Any]) -> None:
    msg.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))

    state.messages.append(msg)

    if should_persist_message(msg):
        save_messages()

    dead = []
    for q in state.subscribers:
        try:
            q.put_nowait(msg)
        except Exception:
            dead.append(q)

    for q in dead:
        if q in state.subscribers:
            state.subscribers.remove(q)


def broadcast(role: str, text: str, images: list[dict[str, str]] | None = None) -> None:
    msg: Dict[str, Any] = {
        "role": role,
        "text": text,
    }

    if images:
        msg["images"] = images

    broadcast_message(msg)
