#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List


TASK_TYPES = {
    "chat_answer",
    "code_edit",
    "code_review",
    "debugging",
    "image_generation",
    "image_editing",
    "translation",
    "document_work",
    "spreadsheet_work",
    "web_research",
    "planning",
    "multi_modal",
}

QUALITY_MODES = {"economy", "balanced", "high_quality"}
ROLE_KEYS = {"router", "planner", "coding", "vision"}


@dataclass
class TaskClassification:
    task_type: str
    secondary_task_types: List[str] = field(default_factory=list)
    requested_artifact: str = "answer"
    required_capabilities: List[str] = field(default_factory=list)
    risk: str = "low"
    complexity: str = "low"
    confidence: float = 0.5
    needs_images: bool = False
    needs_tools: bool = False

    def normalized(self) -> "TaskClassification":
        task_type = self.task_type if self.task_type in TASK_TYPES else "chat_answer"
        secondary = [
            task for task in self.secondary_task_types
            if task in TASK_TYPES and task != task_type
        ]
        return TaskClassification(
            task_type=task_type,
            secondary_task_types=secondary,
            requested_artifact=self.requested_artifact or "answer",
            required_capabilities=clean_string_list(self.required_capabilities),
            risk=normalize_level(self.risk, {"low", "medium", "high"}, "low"),
            complexity=normalize_level(self.complexity, {"low", "medium", "high"}, "low"),
            confidence=max(0.0, min(1.0, float_or_default(self.confidence, 0.5))),
            needs_images=bool(self.needs_images),
            needs_tools=bool(self.needs_tools),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.normalized())


@dataclass
class RoutingDecision:
    mode: str
    selected_model: str
    selected_role: str
    classification_source: str
    reason: str
    fallback_used: bool = False
    classification: TaskClassification | None = None
    debug_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.classification:
            data["classification"] = self.classification.to_dict()
        return data


def clean_string_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []

    result = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def float_or_default(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_level(value: object, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def regex_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def deterministic_classify(task: str, task_images: List[str] | None = None) -> TaskClassification:
    text = task.strip()
    lower = text.lower()
    task_images = task_images or []
    has_images = bool(task_images)

    translation_patterns = (
        r"\btranslate\b",
        r"\btranslation\b",
        r"\binto (german|english|french|spanish|italian|portuguese|dutch|polish|japanese|chinese)\b",
        r"\bauf deutsch\b",
        r"\bin deutsch\b",
    )
    image_edit_terms = (
        "edit this photo",
        "edit this image",
        "make it brighter",
        "make it darker",
        "remove the background",
        "crop this",
        "retouch",
        "upscale",
    )
    image_generation_terms = (
        "generate an image",
        "create an image",
        "draw ",
        "paint ",
        "make a logo",
        "illustration of",
        "picture of",
    )
    code_terms = (
        "fix this",
        "fix the",
        "bug",
        "code",
        "react",
        "component",
        "function",
        "class",
        "refactor",
        "run tests",
        "test failure",
        "write a file",
        "create a file",
        "html file",
        "css",
        "javascript",
        "python",
        "dockerfile",
        "localhost",
        "browser",
        "ui",
        "web app",
        "landing page",
        "dashboard",
    )
    planning_terms = (
        "plan",
        "architecture",
        "design",
        "approach",
        "strategy",
        "multi-step",
        "workflow",
        "debug why",
        "investigate",
    )
    web_terms = (
        "look up",
        "search the web",
        "browse",
        "fetch",
        "website",
        "url",
        "http://",
        "https://",
    )

    if has_images and contains_any(lower, image_edit_terms):
        return TaskClassification(
            task_type="image_editing",
            requested_artifact="image_analysis_or_edit_instruction",
            required_capabilities=["vision"],
            risk="medium",
            complexity="medium",
            confidence=0.95,
            needs_images=True,
            needs_tools=True,
        )

    if has_images:
        return TaskClassification(
            task_type="multi_modal",
            secondary_task_types=["image_editing"],
            requested_artifact="answer",
            required_capabilities=["vision"],
            risk="medium",
            complexity="medium",
            confidence=0.9,
            needs_images=True,
            needs_tools=False,
        )

    if contains_any(lower, image_generation_terms):
        secondary = ["code_edit"] if contains_any(lower, ("html", "page", "website", "landing page")) else []
        task_type = "code_edit" if secondary else "image_generation"
        return TaskClassification(
            task_type=task_type,
            secondary_task_types=["image_generation"] if task_type == "code_edit" else [],
            requested_artifact="visual_asset" if task_type == "image_generation" else "web_page",
            required_capabilities=["visual_generation"] + (["file_editing"] if task_type == "code_edit" else []),
            risk="medium",
            complexity="medium",
            confidence=0.88,
            needs_images=False,
            needs_tools=task_type == "code_edit",
        )

    if regex_any(lower, translation_patterns) and not contains_any(lower, code_terms):
        return TaskClassification(
            task_type="translation",
            requested_artifact="translated_text",
            required_capabilities=["translation"],
            risk="low",
            complexity="low",
            confidence=0.92,
            needs_images=False,
            needs_tools=False,
        )

    if contains_any(lower, code_terms):
        secondary = ["image_generation"] if contains_any(lower, image_generation_terms + ("hero image", "image")) else []
        return TaskClassification(
            task_type="code_edit",
            secondary_task_types=secondary,
            requested_artifact="modified_files",
            required_capabilities=["code_reasoning", "file_editing", "tool_use"],
            risk="medium",
            complexity="medium",
            confidence=0.9,
            needs_images=False,
            needs_tools=True,
        )

    if contains_any(lower, web_terms):
        return TaskClassification(
            task_type="web_research",
            requested_artifact="answer",
            required_capabilities=["web_fetch", "tool_use"],
            risk="medium",
            complexity="medium",
            confidence=0.86,
            needs_images=False,
            needs_tools=True,
        )

    if contains_any(lower, planning_terms):
        return TaskClassification(
            task_type="planning",
            requested_artifact="plan",
            required_capabilities=["reasoning"],
            risk="medium",
            complexity="medium",
            confidence=0.82,
            needs_images=False,
            needs_tools=False,
        )

    if (
        len(lower) <= 120
        and not contains_any(lower, ("does not work", "doesn't work", "broken", "error", "failing"))
        and (
            lower in {"hi", "hello", "hey", "thanks", "thank you"}
            or lower.endswith("?")
            or contains_any(lower, ("what do you think", "explain", "why", "how would"))
        )
    ):
        return TaskClassification(
            task_type="chat_answer",
            requested_artifact="answer",
            required_capabilities=[],
            risk="low",
            complexity="low",
            confidence=0.88,
            needs_images=False,
            needs_tools=False,
        )

    return TaskClassification(
        task_type="chat_answer",
        requested_artifact="answer",
        required_capabilities=[],
        risk="low",
        complexity="low",
        confidence=0.72,
        needs_images=False,
        needs_tools=False,
    )


def build_classifier_prompt(
    task: str,
    conversation_context: List[Dict[str, object]] | None = None,
    task_images: List[str] | None = None,
) -> str:
    return f"""
Classify the user's request for a local Ollama agent. Do not solve the task.

Return exactly one JSON object with these keys:
- task_type: one of {sorted(TASK_TYPES)}
- secondary_task_types: list of task types
- requested_artifact: short snake_case description
- required_capabilities: list of short snake_case capabilities
- risk: low, medium, or high
- complexity: low, medium, or high
- confidence: number from 0 to 1
- needs_images: boolean
- needs_tools: boolean

Important:
- Classify by the artifact the user wants at the end.
- A website/page/app request with an image is primarily code_edit, with image_generation secondary.
- Code/file/test/debug/browser work needs tools.
- Uploaded image filenames mean the task may need vision.

Uploaded image filenames:
{json.dumps(task_images or [], ensure_ascii=False)}

Recent context:
{json.dumps(conversation_context or [], ensure_ascii=False)[:6000]}

User request:
{task}
""".strip()


def parse_classifier_response(text: str) -> TaskClassification:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(cleaned[start:end + 1])

    return TaskClassification(
        task_type=str(data.get("task_type", "chat_answer")),
        secondary_task_types=clean_string_list(data.get("secondary_task_types", [])),
        requested_artifact=str(data.get("requested_artifact", "answer")),
        required_capabilities=clean_string_list(data.get("required_capabilities", [])),
        risk=str(data.get("risk", "low")),
        complexity=str(data.get("complexity", "low")),
        confidence=float_or_default(data.get("confidence"), 0.5),
        needs_images=bool(data.get("needs_images", False)),
        needs_tools=bool(data.get("needs_tools", False)),
    ).normalized()


def choose_role(classification: TaskClassification, quality_mode: str) -> tuple[str, str]:
    classification = classification.normalized()
    quality_mode = normalize_level(quality_mode, QUALITY_MODES, "balanced")

    if classification.needs_images or classification.task_type in {"image_editing", "multi_modal"}:
        return "vision", "image or multimodal task needs vision"

    if classification.task_type in {"code_edit", "code_review", "debugging"}:
        return "coding", "code, file, or debugging task needs coding model"

    if classification.needs_tools and "code_reasoning" in classification.required_capabilities:
        return "coding", "tool task includes code reasoning"

    if classification.task_type == "planning":
        return "planner", "planning task benefits from reasoning model"

    if classification.task_type == "web_research":
        return ("planner", "research task benefits from planning model") if quality_mode != "economy" else ("router", "economy mode uses router for low-risk research")

    if classification.task_type == "translation":
        return ("router", "translation can use lightweight router role") if quality_mode == "economy" else ("default", "translation uses default model in balanced quality")

    if classification.task_type == "image_generation":
        return "default", "local image generation provider is not configured in v1"

    if quality_mode == "high_quality" and (
        classification.complexity != "low" or classification.risk != "low"
    ):
        return "planner", "high quality mode escalates non-trivial tasks"

    if classification.confidence < 0.6 and quality_mode != "economy":
        return "planner", "low classification confidence escalates upward"

    return "default", "plain response can use default model"


def route_task(
    task: str,
    default_model: str,
    roles: Dict[str, str] | None = None,
    quality_mode: str = "balanced",
    enabled: bool = True,
    task_images: List[str] | None = None,
    conversation_context: List[Dict[str, object]] | None = None,
    classifier: Callable[[str, str], str] | None = None,
) -> RoutingDecision:
    roles = roles or {}
    quality_mode = normalize_level(quality_mode, QUALITY_MODES, "balanced")
    debug_details: Dict[str, Any] = {
        "enabled": enabled,
        "quality_mode": quality_mode,
        "default_model": default_model,
        "configured_roles": dict(roles),
        "task_preview": task[:240],
        "task_image_count": len(task_images or []),
        "classifier_threshold": 0.8,
    }

    if not enabled:
        classification = deterministic_classify(task, task_images).normalized()
        debug_details["deterministic_classification"] = classification.to_dict()
        return RoutingDecision(
            mode=quality_mode,
            selected_model=default_model,
            selected_role="default",
            classification_source="disabled",
            reason="routing disabled",
            fallback_used=False,
            classification=classification,
            debug_details=debug_details,
        )

    classification = deterministic_classify(task, task_images).normalized()
    source = "deterministic"
    debug_details["deterministic_classification"] = classification.to_dict()
    debug_details["classifier_attempted"] = False

    if classification.confidence < 0.8 and classifier is not None:
        router_model = roles.get("router") or default_model
        prompt = build_classifier_prompt(task, conversation_context, task_images)
        debug_details["classifier_attempted"] = True
        debug_details["classifier_model"] = router_model
        try:
            model_response = classifier(prompt, router_model)
            debug_details["classifier_response_preview"] = model_response[:2000]
            classified = parse_classifier_response(model_response)
            debug_details["model_classification"] = classified.to_dict()
            if classified.confidence >= classification.confidence:
                classification = classified
                source = "model"
                debug_details["model_classification_used"] = True
            else:
                debug_details["model_classification_used"] = False
        except Exception:
            source = "deterministic_after_classifier_error"
            debug_details["classifier_error"] = True

    selected_role, reason = choose_role(classification, quality_mode)
    selected_model = default_model if selected_role == "default" else roles.get(selected_role, "")
    fallback_used = False

    if not selected_model:
        selected_model = default_model
        fallback_used = selected_role != "default"

    debug_details["final_classification"] = classification.to_dict()
    debug_details["selected_role"] = selected_role
    debug_details["selected_model"] = selected_model
    debug_details["fallback_used"] = fallback_used
    debug_details["reason"] = reason

    return RoutingDecision(
        mode=quality_mode,
        selected_model=selected_model,
        selected_role=selected_role,
        classification_source=source,
        reason=reason,
        fallback_used=fallback_used,
        classification=classification,
        debug_details=debug_details,
    )
