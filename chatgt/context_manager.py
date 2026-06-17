#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, Dict, List


def estimate_tokens(text: str) -> int:
    """
    Fast conservative token estimate for budgeting.

    Ollama reports exact prompt_eval_count after a request, but the app needs a
    preflight budget before sending the prompt. Four chars per token is a
    practical approximation for English/code mixed prompts.
    """
    return max(1, (len(text) + 3) // 4)


def clamp_int(value: object, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default

    return max(minimum, min(maximum, parsed))


class ContextManager:
    def __init__(
        self,
        recent_message_limit: int,
        recent_message_char_limit: int,
        target_tokens: int,
    ):
        self.recent_message_limit = recent_message_limit
        self.recent_message_char_limit = recent_message_char_limit
        self.target_tokens = target_tokens

    def compact_message(self, msg: Dict[str, Any]) -> Dict[str, Any] | None:
        role = str(msg.get("role", ""))
        text = str(msg.get("text", ""))

        if role not in {"user", "agent"}:
            return None

        if not text.strip():
            return None

        return {
            "role": role,
            "text": text[:self.recent_message_char_limit],
            "timestamp": msg.get("timestamp"),
        }

    def recent_conversation(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        context = []

        for msg in messages:
            compact = self.compact_message(msg)
            if compact:
                context.append(compact)

        return context[-self.recent_message_limit:]

    def compact_task_history(
        self,
        history: List[Dict[str, object]],
        max_items: int = 12,
        max_observation_chars: int = 3000,
    ) -> List[Dict[str, object]]:
        compact_history = []

        for item in history[-max_items:]:
            compact_item = dict(item)
            observation = compact_item.get("observation")

            if isinstance(observation, dict):
                compact_observation = dict(observation)
                content = compact_observation.get("content")

                if isinstance(content, str) and len(content) > max_observation_chars:
                    compact_observation["content"] = content[:max_observation_chars]
                    compact_observation["content_truncated_for_history"] = True

                compact_item["observation"] = compact_observation

            compact_history.append(compact_item)

        return compact_history

    def context_status(self, memory_summary: str, recent_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        recent_text = "\n".join(str(item.get("text", "")) for item in recent_context)

        return {
            "target_tokens": self.target_tokens,
            "memory_summary_chars": len(memory_summary),
            "memory_summary_estimated_tokens": estimate_tokens(memory_summary),
            "recent_messages": len(recent_context),
            "recent_context_chars": len(recent_text),
            "recent_context_estimated_tokens": estimate_tokens(recent_text) if recent_text else 0,
        }
