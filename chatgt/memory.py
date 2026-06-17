#!/usr/bin/env python3

from typing import Any, Dict, List

import app_state as state
from persistence import load_memory_state, save_memory_state, save_messages


def get_recent_conversation_context() -> List[Dict[str, Any]]:
    """
    Returns recent user/agent messages for conversational continuity.

    This is separate from long-term memory. It lets the model understand
    follow-ups like "make it rhyme", "change that", "continue", etc.
    """
    return state.context_manager.recent_conversation(state.messages)


def compact_message_for_memory(msg: Dict[str, Any]) -> Dict[str, Any] | None:
    role = str(msg.get("role", ""))
    text = str(msg.get("text", ""))

    if role in {"progress", "status"}:
        return None

    if not text.strip():
        return None

    return {
        "role": role,
        "text": text[:3000],
        "timestamp": msg.get("timestamp"),
    }


def maybe_summarize_memory() -> None:
    """
    Updates long-term memory without deleting visible chat history.

    The UI keeps all messages in .agent_sessions.json.
    Only older messages are summarized into .agent_memory_summary.txt.
    .agent_memory_state.json prevents repeatedly summarizing the same messages.
    """
    if len(state.messages) <= state.SUMMARIZE_AFTER_MESSAGES:
        return

    memory_state = load_memory_state()
    summarized_until_index = int(memory_state.get("summarized_until_index", 0))

    summarize_until_index = max(0, len(state.messages) - state.RECENT_MESSAGES_TO_KEEP)

    if summarize_until_index <= summarized_until_index:
        return

    messages_to_summarize = state.messages[summarized_until_index:summarize_until_index]

    memory_candidates = []
    for msg in messages_to_summarize:
        compact = compact_message_for_memory(msg)
        if compact:
            memory_candidates.append(compact)

    if not memory_candidates:
        memory_state["summarized_until_index"] = summarize_until_index
        save_memory_state(memory_state)
        return

    try:
        existing_summary = state.agent.load_memory_summary()
        updated_summary = state.agent.summarize_conversation_memory(
            existing_summary,
            memory_candidates,
        )
        state.agent.save_memory_summary(updated_summary)

        memory_state["summarized_until_index"] = summarize_until_index
        save_memory_state(memory_state)

        save_messages()

        print(
            f"Updated memory summary from {len(memory_candidates)} messages. "
            f"Kept all {len(state.messages)} visible messages."
        )

    except Exception as exc:
        print(f"Memory summarization failed: {exc}")


def prepare_context_for_next_task() -> Dict[str, Any]:
    """
    Summarize stale visible messages before prompt construction.

    The visible UI transcript is still retained. This only updates the compact
    memory layer and then returns the recent conversation slice that should be
    included verbatim in the next model request.
    """
    maybe_summarize_memory()

    recent_context = get_recent_conversation_context()
    summary = state.agent.load_memory_summary()

    return {
        "recent_context": recent_context,
        "status": state.context_manager.context_status(summary, recent_context),
    }
