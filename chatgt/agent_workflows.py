#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List


WorkflowProgress = Callable[[Dict[str, object]], None] | None
WORKFLOW_CONTEXT_TEXT_LIMIT = 500
WORKFLOW_FILE = ".agent_workflows.json"
WORKFLOW_ROLES = {"default", "router", "planner", "coding", "vision"}


@dataclass
class StepResult:
    step_id: str
    status: str
    summary: str
    findings: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    raw_output: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class WorkflowStep:
    id: str
    type: str = "agentic_task"
    role: str = "default"
    prompt: str = ""
    condition: str = ""
    tool_guidance: str = ""
    max_iterations: int = 8


@dataclass
class WorkflowLoop:
    id: str
    steps: List[WorkflowStep]
    until: str
    max_iterations: int = 2


@dataclass
class WorkflowDefinition:
    id: str
    name: str
    description: str
    steps: List[WorkflowStep | WorkflowLoop]


@dataclass
class WorkflowRun:
    task: str
    workflow_id: str
    status: str = "running"
    results: Dict[str, StepResult] = field(default_factory=dict)
    timeline: List[Dict[str, object]] = field(default_factory=list)


def workflow_catalog() -> List[WorkflowDefinition]:
    return [
        WorkflowDefinition(
            id="code_task",
            name="Code task",
            description="Plan, implement, check, and fix a code task.",
            steps=[
                WorkflowStep(
                    id="plan",
                    role="planner",
                    prompt="Create a concise implementation plan for this coding task.",
                    tool_guidance="Strongly prefer list_files and read_file to inspect relevant files before planning. Do not edit files in this step unless inspection absolutely requires creating a temporary artifact.",
                    max_iterations=4,
                ),
                WorkflowStep(
                    id="implement",
                    role="coding",
                    prompt="Implement the requested coding task. Use the plan and prior workflow context.",
                    tool_guidance="Use read_file before write_file. Use write_file for focused changes. Use run_command or browser tools when they help verify the implementation.",
                    max_iterations=8,
                ),
                WorkflowLoop(
                    id="test_fix_loop",
                    max_iterations=3,
                    until="visual_check.status == \"passed\"",
                    steps=[
                        WorkflowStep(
                            id="check",
                            role="coding",
                            prompt="Run or perform the most appropriate available checks for the task. If no executable checks are possible, inspect the result and report passed only when it is reasonably verified.",
                            tool_guidance="Strongly prefer run_command, read_file, browser_open, browser_snapshot, or browser_screenshot. For HTML, canvas, animation, game, or visual UI tasks, browser DOM text/title alone is not enough; leave visual judgment to the visual_check step. Do not edit files in this step.",
                            max_iterations=4,
                        ),
                        WorkflowStep(
                            id="visual_check",
                            role="vision",
                            condition="check.status == \"passed\"",
                            prompt="Decide whether visual/browser verification is required for this task. If the task is not visual, UI, HTML, CSS, canvas, animation, game, page, layout, or screenshot-related, return passed with a summary that no visual check is required. If it is visual, open the result in the browser, take a browser_screenshot, inspect the screenshot evidence, and return passed only if the visible result matches the task. If no screenshot evidence is available for a visual task, return needs_changes.",
                            tool_guidance="For visual tasks, use browser_open and browser_screenshot. Prefer send_image when screenshot evidence should be attached. Use the configured vision role for visual judgment. Do not pass visual tasks from DOM title/text alone. Do not edit files in this step.",
                            max_iterations=5,
                        ),
                        WorkflowStep(
                            id="fix",
                            role="coding",
                            condition="visual_check.status != \"passed\"",
                            prompt="Fix the issues reported by the check or visual_check step. Do not make unrelated changes.",
                            tool_guidance="Use the check and visual_check findings as the source of truth. Read relevant files before writing. Use write_file only for focused fixes.",
                            max_iterations=8,
                        ),
                    ],
                ),
            ],
        ),
        WorkflowDefinition(
            id="frontend_visual",
            name="Frontend visual",
            description="Plan, implement, capture screenshots, analyze, and improve frontend work.",
            steps=[
                WorkflowStep(
                    id="plan",
                    role="planner",
                    prompt="Create a concise frontend implementation and visual verification plan.",
                    tool_guidance="Inspect existing files with list_files/read_file. If a page already exists, read it before planning.",
                    max_iterations=4,
                ),
                WorkflowStep(
                    id="implement",
                    role="coding",
                    prompt="Implement the frontend/UI task. Prefer a complete runnable artifact when creating a new page.",
                    tool_guidance="Use read_file and write_file for UI changes. Use browser tools if they help identify the target page.",
                    max_iterations=8,
                ),
                WorkflowLoop(
                    id="visual_polish_loop",
                    max_iterations=2,
                    until="analyze.status == \"passed\"",
                    steps=[
                        WorkflowStep(
                            id="screenshot",
                            role="coding",
                            prompt="Open or inspect the resulting UI with the browser tools and save a screenshot artifact. If no UI can be opened, explain why.",
                            tool_guidance="Strongly prefer browser_open, browser_snapshot, browser_screenshot, and send_image. For visual output, capture a screenshot; DOM text/title alone is not visual verification. Do not edit files in this step.",
                            max_iterations=4,
                        ),
                        WorkflowStep(
                            id="analyze",
                            role="vision",
                            prompt="Analyze the latest UI/screenshot result for visual issues such as overlap, unreadable text, broken layout, missing content, or obvious polish problems.",
                            tool_guidance="Use browser_snapshot, browser_screenshot, send_image, and read_file as needed. Do not edit files in this step.",
                            max_iterations=4,
                        ),
                        WorkflowStep(
                            id="improve",
                            role="coding",
                            condition="analyze.status != \"passed\"",
                            prompt="Improve the implementation based on the visual analysis findings. Keep edits focused.",
                            tool_guidance="Use the visual findings as the source of truth. Read affected files before writing and verify with browser tools when practical.",
                            max_iterations=8,
                        ),
                    ],
                ),
            ],
        ),
        WorkflowDefinition(
            id="debugging",
            name="Debugging",
            description="Inspect, hypothesize, fix, verify, and retry a debugging task.",
            steps=[
                WorkflowStep(
                    id="inspect",
                    role="coding",
                    prompt="Inspect the problem, gather relevant files/output, and identify likely causes without broad unrelated edits.",
                    tool_guidance="Strongly prefer list_files, read_file, run_command, fetch_url, and browser tools as appropriate. Do not edit files in this step.",
                    max_iterations=6,
                ),
                WorkflowStep(
                    id="hypothesis",
                    role="planner",
                    prompt="Summarize the likely root cause and the smallest fix strategy.",
                    tool_guidance="Use prior inspect results, and read additional files if needed. Do not edit files in this step.",
                    max_iterations=4,
                ),
                WorkflowStep(
                    id="implement_fix",
                    role="coding",
                    prompt="Implement the smallest practical fix for the debugging task.",
                    tool_guidance="Use read_file before write_file. Keep fixes focused on the diagnosed issue.",
                    max_iterations=8,
                ),
                WorkflowLoop(
                    id="verify_fix_loop",
                    max_iterations=3,
                    until="verify.status == \"passed\"",
                    steps=[
                        WorkflowStep(
                            id="verify",
                            role="coding",
                            prompt="Verify whether the bug is fixed using the best available checks or reproduction steps.",
                            tool_guidance="Strongly prefer run_command, browser tools, and read_file. Do not edit files in this step.",
                            max_iterations=4,
                        ),
                        WorkflowStep(
                            id="repair",
                            role="coding",
                            condition="verify.status != \"passed\"",
                            prompt="Use the verification failure findings to repair the fix. Avoid unrelated changes.",
                            tool_guidance="Use verification findings as the source of truth. Read before writing. Keep repairs focused.",
                            max_iterations=8,
                        ),
                    ],
                ),
            ],
        ),
    ]


def workflow_options() -> List[Dict[str, object]]:
    return workflow_options_for_workdir(None)


def workflow_options_for_workdir(workdir: str | None = None) -> List[Dict[str, object]]:
    options: List[Dict[str, object]] = [
        {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "editable": False,
            "source": "built-in",
        }
        for workflow in workflow_catalog()
    ]

    if workdir:
        options.extend(
            {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "editable": True,
                "source": "custom",
            }
            for workflow in load_custom_workflows(workdir)
        )

    return options


def built_in_workflow_ids() -> set[str]:
    return {workflow.id for workflow in workflow_catalog()}


def workflow_item_to_dict(item: WorkflowStep | WorkflowLoop) -> Dict[str, object]:
    if isinstance(item, WorkflowLoop):
        return {
            "kind": "loop",
            "id": item.id,
            "until": item.until,
            "max_iterations": item.max_iterations,
            "steps": [workflow_item_to_dict(step) for step in item.steps],
        }

    return {
        "kind": "step",
        "id": item.id,
        "type": item.type,
        "role": item.role,
        "prompt": item.prompt,
        "condition": item.condition,
        "tool_guidance": item.tool_guidance,
        "max_iterations": item.max_iterations,
    }


def workflow_to_dict(workflow: WorkflowDefinition) -> Dict[str, object]:
    return {
        "id": workflow.id,
        "name": workflow.name,
        "description": workflow.description,
        "steps": [workflow_item_to_dict(item) for item in workflow.steps],
    }


def workflow_item_from_dict(data: Dict[str, object]) -> WorkflowStep | WorkflowLoop:
    kind = str(data.get("kind", "") or "").strip().lower()
    if kind == "loop" or "until" in data:
        steps_data = data.get("steps", [])
        if not isinstance(steps_data, list):
            raise ValueError(f"Loop {data.get('id', '')} steps must be a list")
        return WorkflowLoop(
            id=str(data.get("id", "") or "").strip(),
            until=str(data.get("until", "") or "").strip(),
            max_iterations=int(data.get("max_iterations", 2) or 2),
            steps=[
                workflow_item_from_dict(step)
                for step in steps_data
                if isinstance(step, dict)
            ],
        )

    return WorkflowStep(
        id=str(data.get("id", "") or "").strip(),
        type=str(data.get("type", "agentic_task") or "agentic_task").strip(),
        role=str(data.get("role", "default") or "default").strip(),
        prompt=str(data.get("prompt", "") or ""),
        condition=str(data.get("condition", "") or "").strip(),
        tool_guidance=str(data.get("tool_guidance", "") or ""),
        max_iterations=int(data.get("max_iterations", 8) or 8),
    )


def workflow_from_dict(data: Dict[str, object]) -> WorkflowDefinition:
    steps_data = data.get("steps", [])
    if not isinstance(steps_data, list):
        raise ValueError("Workflow steps must be a list")
    return WorkflowDefinition(
        id=str(data.get("id", "") or "").strip(),
        name=str(data.get("name", "") or "").strip(),
        description=str(data.get("description", "") or ""),
        steps=[
            workflow_item_from_dict(item)
            for item in steps_data
            if isinstance(item, dict)
        ],
    )


def custom_workflow_path(workdir: str) -> str:
    return os.path.join(workdir, WORKFLOW_FILE)


def load_custom_workflows(workdir: str) -> List[WorkflowDefinition]:
    path = custom_workflow_path(workdir)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    items = data.get("workflows", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []

    workflows: List[WorkflowDefinition] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            workflow = workflow_from_dict(item)
            validate_workflow(workflow, reserved_ids=built_in_workflow_ids())
        except ValueError:
            continue
        if workflow.id in seen:
            continue
        seen.add(workflow.id)
        workflows.append(workflow)

    return workflows


def save_custom_workflows(workdir: str, workflows: List[WorkflowDefinition]) -> None:
    os.makedirs(workdir, exist_ok=True)
    with open(custom_workflow_path(workdir), "w", encoding="utf-8") as f:
        json.dump(
            {"workflows": [workflow_to_dict(workflow) for workflow in workflows]},
            f,
            indent=2,
        )
        f.write("\n")


def save_custom_workflow(workdir: str, workflow: WorkflowDefinition, replacing_id: str | None = None) -> WorkflowDefinition:
    validate_workflow(workflow, reserved_ids=built_in_workflow_ids())
    workflows = load_custom_workflows(workdir)
    target_id = replacing_id or workflow.id

    if replacing_id and replacing_id in built_in_workflow_ids():
        raise ValueError(f"Cannot update built-in workflow: {replacing_id}")
    if any(existing.id == workflow.id and existing.id != target_id for existing in workflows):
        raise ValueError(f"Custom workflow already exists: {workflow.id}")

    replaced = False
    for index, existing in enumerate(workflows):
        if existing.id == target_id:
            workflows[index] = workflow
            replaced = True
            break

    if not replaced:
        workflows.append(workflow)

    save_custom_workflows(workdir, workflows)
    return workflow


def delete_custom_workflow(workdir: str, workflow_id: str) -> bool:
    workflows = load_custom_workflows(workdir)
    kept = [workflow for workflow in workflows if workflow.id != workflow_id]
    if len(kept) == len(workflows):
        return False
    save_custom_workflows(workdir, kept)
    return True


def get_workflow(workflow_id: str, workdir: str | None = None) -> WorkflowDefinition | None:
    for workflow in workflow_catalog():
        if workflow.id == workflow_id:
            return workflow
    if workdir:
        for workflow in load_custom_workflows(workdir):
            if workflow.id == workflow_id:
                return workflow
    return None


def get_workflow_detail(workflow_id: str, workdir: str | None = None) -> Dict[str, object] | None:
    for workflow in workflow_catalog():
        if workflow.id == workflow_id:
            data = workflow_to_dict(workflow)
            data.update({"editable": False, "source": "built-in"})
            return data
    if workdir:
        for workflow in load_custom_workflows(workdir):
            if workflow.id == workflow_id:
                data = workflow_to_dict(workflow)
                data.update({"editable": True, "source": "custom"})
                return data
    return None


def clone_workflow(workdir: str, workflow_id: str, new_id: str | None = None, new_name: str | None = None) -> WorkflowDefinition:
    source = get_workflow(workflow_id, workdir)
    if source is None:
        raise ValueError(f"Unknown workflow: {workflow_id}")

    existing_ids = built_in_workflow_ids() | {workflow.id for workflow in load_custom_workflows(workdir)}
    base_id = re.sub(r"[^A-Za-z0-9_-]+", "_", (new_id or f"{source.id}_copy")).strip("_")
    candidate = base_id or "custom_workflow"
    if candidate in existing_ids:
        suffix = 2
        while f"{candidate}_{suffix}" in existing_ids:
            suffix += 1
        candidate = f"{candidate}_{suffix}"

    clone = workflow_from_dict(workflow_to_dict(source))
    clone.id = candidate
    clone.name = new_name or f"{source.name} copy"
    return save_custom_workflow(workdir, clone)


def workflow_validation_errors(workflow: WorkflowDefinition, reserved_ids: set[str] | None = None) -> List[str]:
    try:
        validate_workflow(workflow, reserved_ids=reserved_ids)
    except ValueError as exc:
        return [str(exc)]
    return []


def validate_workflow(workflow: WorkflowDefinition, reserved_ids: set[str] | None = None) -> None:
    if not workflow.id:
        raise ValueError("Workflow is missing id")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", workflow.id):
        raise ValueError("Workflow id may only contain letters, numbers, underscores, and hyphens")
    if reserved_ids and workflow.id in reserved_ids:
        raise ValueError(f"Workflow id is reserved: {workflow.id}")
    if not workflow.name:
        raise ValueError("Workflow is missing name")
    if not workflow.steps:
        raise ValueError("Workflow must contain at least one step")

    seen = set()

    def visit_step(step: WorkflowStep) -> None:
        if not step.id:
            raise ValueError("Workflow step is missing id")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", step.id):
            raise ValueError(f"Invalid step id: {step.id}")
        if step.id in seen:
            raise ValueError(f"Duplicate step id: {step.id}")
        if step.type != "agentic_task":
            raise ValueError(f"Invalid step type for {step.id}: {step.type}")
        if step.role not in WORKFLOW_ROLES:
            raise ValueError(f"Invalid role for step {step.id}: {step.role}")
        if step.max_iterations < 1 or step.max_iterations > 24:
            raise ValueError(f"Invalid max_iterations for step {step.id}")
        seen.add(step.id)

    for item in workflow.steps:
        if isinstance(item, WorkflowLoop):
            if not item.id:
                raise ValueError("Workflow loop is missing id")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", item.id):
                raise ValueError(f"Invalid loop id: {item.id}")
            if item.max_iterations < 1 or item.max_iterations > 10:
                raise ValueError(f"Invalid max_iterations for loop {item.id}")
            if not item.steps:
                raise ValueError(f"Loop {item.id} must contain at least one step")
            for step in item.steps:
                visit_step(step)
        else:
            visit_step(item)

    known = set(seen)

    def validate_condition(condition: str) -> None:
        if not condition:
            return
        referenced = condition.split(".", 1)[0].strip()
        if referenced and referenced not in known:
            raise ValueError(f"Condition references unknown step: {referenced}")
        evaluate_condition(condition, {})

    for item in workflow.steps:
        if isinstance(item, WorkflowLoop):
            validate_condition(item.until)
            for step in item.steps:
                validate_condition(step.condition)
        else:
            validate_condition(item.condition)


def evaluate_condition(condition: str, results: Dict[str, StepResult]) -> bool:
    condition = condition.strip()
    if not condition:
        return True

    match = re.fullmatch(
        r"([A-Za-z0-9_-]+)\.(status|findings_count)\s*(==|!=|>=|<=|>|<)\s*(?:\"([^\"]*)\"|(\d+))",
        condition,
    )
    if not match:
        raise ValueError(f"Unsupported workflow condition: {condition}")

    step_id, field_name, op, text_value, number_value = match.groups()
    result = results.get(step_id)
    if result is None:
        return False

    if field_name == "status":
        left: object = result.status
        right: object = text_value or ""
    else:
        left = len(result.findings)
        right = int(number_value or 0)

    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == ">":
        return int(left) > int(right)
    if op == "<":
        return int(left) < int(right)
    if op == ">=":
        return int(left) >= int(right)
    if op == "<=":
        return int(left) <= int(right)

    raise ValueError(f"Unsupported workflow operator: {op}")


def parse_step_result(step_id: str, raw_output: str, require_structured: bool = True) -> StepResult:
    text = raw_output.strip()
    data: Dict[str, object] | None = None

    cleaned = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                data = None

    if data:
        findings = data.get("findings", [])
        artifacts = data.get("artifacts", [])
        return StepResult(
            step_id=step_id,
            status=normalize_status(data.get("status")),
            summary=str(data.get("summary", "") or text[:500]),
            findings=[str(item) for item in findings] if isinstance(findings, list) else [],
            artifacts=[str(item) for item in artifacts] if isinstance(artifacts, list) else [],
            raw_output=raw_output,
        )

    lowered = text.lower()
    status = "needs_changes" if require_structured else "passed"
    if any(term in lowered for term in ("failed", "error", "blocked", "could not", "incomplete")):
        status = "failed"
    if "needs_changes" in lowered or "needs changes" in lowered:
        status = "needs_changes"

    return StepResult(
        step_id=step_id,
        status=status,
        summary=text[:WORKFLOW_CONTEXT_TEXT_LIMIT] if text else "Step completed.",
        findings=[] if status == "passed" else [
            "Step did not return the required structured JSON result.",
            text[:WORKFLOW_CONTEXT_TEXT_LIMIT],
        ],
        raw_output=raw_output,
    )


def compact_step_result_for_context(result: StepResult) -> Dict[str, object]:
    return {
        "step_id": result.step_id,
        "status": result.status,
        "summary": result.summary[:WORKFLOW_CONTEXT_TEXT_LIMIT],
        "findings": [
            finding[:WORKFLOW_CONTEXT_TEXT_LIMIT]
            for finding in result.findings[:5]
        ],
        "artifacts": result.artifacts[:10],
    }


def normalize_status(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"needed_changes", "need_changes", "needs change", "needed changes"}:
        return "needs_changes"
    if text in {"passed", "failed", "needs_changes", "blocked", "skipped"}:
        return text
    return "passed"


def workflow_has_passing_verification(run: WorkflowRun) -> bool:
    return any(
        step_id in {"check", "verify", "analyze"}
        and result.status == "passed"
        for step_id, result in run.results.items()
    )


def workflow_status(run: WorkflowRun) -> str:
    failed = [
        result for result in run.results.values()
        if result.status in {"failed", "blocked", "needs_changes"}
    ]
    if not failed:
        return "passed"
    if workflow_has_passing_verification(run):
        active_failures = [
            result for result in failed
            if result.step_id not in {"plan", "hypothesis"}
        ]
        if not active_failures:
            return "passed"
        return "passed_with_warnings"
    return "completed_with_issues"


class WorkflowRunner:
    def __init__(self, agent):
        self.agent = agent
        self._current_debug_run_id = ""

    def run(
        self,
        workflow: WorkflowDefinition,
        task: str,
        progress_callback: WorkflowProgress = None,
        conversation_context: List[Dict[str, object]] | None = None,
        task_images: List[str] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> str:
        validate_workflow(workflow)
        run = WorkflowRun(task=task, workflow_id=workflow.id)
        self._current_debug_run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        conversation_context = conversation_context or []
        task_images = task_images or []

        self.emit(progress_callback, "workflow", f"Workflow started: {workflow.name}")
        self.log_full_event("workflow_started", {
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "task": task,
            "task_images": task_images,
        })

        for item in workflow.steps:
            if self.cancelled(cancel_checker, progress_callback, run):
                break

            if isinstance(item, WorkflowLoop):
                self.run_loop(
                    item,
                    run,
                    task,
                    progress_callback,
                    conversation_context,
                    task_images,
                    cancel_checker,
                )
            else:
                self.run_step(
                    item,
                    run,
                    task,
                    progress_callback,
                    conversation_context,
                    task_images,
                    cancel_checker=cancel_checker,
                )

        run.status = "cancelled" if cancel_checker and cancel_checker() else workflow_status(run)

        summary = self.final_summary(workflow, run)
        append_workflow_minimal_log(self.agent, task, workflow, run, summary, self._current_debug_run_id)
        self.log_full_event("workflow_finished", {
            "workflow_id": workflow.id,
            "status": run.status,
            "summary": summary,
            "results": {
                step_id: result.to_dict()
                for step_id, result in run.results.items()
            },
        })
        return summary

    def run_loop(
        self,
        loop: WorkflowLoop,
        run: WorkflowRun,
        task: str,
        progress_callback: WorkflowProgress,
        conversation_context: List[Dict[str, object]],
        task_images: List[str],
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None:
        self.emit(progress_callback, "workflow", f"Loop started: {loop.id}")
        self.log_full_event("loop_started", {
            "loop_id": loop.id,
            "until": loop.until,
            "max_iterations": loop.max_iterations,
        })

        for iteration in range(1, loop.max_iterations + 1):
            if self.cancelled(cancel_checker, progress_callback, run):
                return

            self.emit(progress_callback, "workflow", f"Loop {loop.id} iteration {iteration}/{loop.max_iterations}")
            self.log_full_event("loop_iteration_started", {
                "loop_id": loop.id,
                "iteration": iteration,
            })

            for step in loop.steps:
                self.run_step(
                    step,
                    run,
                    task,
                    progress_callback,
                    conversation_context,
                    task_images,
                    loop_iteration=iteration,
                    cancel_checker=cancel_checker,
                )

                if self.cancelled(cancel_checker, progress_callback, run):
                    return

            if evaluate_condition(loop.until, run.results):
                self.emit(progress_callback, "workflow", f"Loop {loop.id} condition met")
                self.log_full_event("loop_condition_met", {
                    "loop_id": loop.id,
                    "iteration": iteration,
                    "until": loop.until,
                })
                return

        self.emit(progress_callback, "workflow", f"Loop {loop.id} reached max iterations")
        self.log_full_event("loop_max_iterations_reached", {
            "loop_id": loop.id,
            "max_iterations": loop.max_iterations,
            "until": loop.until,
        })

    def run_step(
        self,
        step: WorkflowStep,
        run: WorkflowRun,
        task: str,
        progress_callback: WorkflowProgress,
        conversation_context: List[Dict[str, object]],
        task_images: List[str],
        loop_iteration: int | None = None,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> StepResult:
        if cancel_checker and cancel_checker():
            result = StepResult(step.id, "blocked", "Skipped because task was aborted.")
            run.results[step.id] = result
            self.emit(progress_callback, "workflow", f"Aborted before step {step.id}")
            self.log_full_event("step_cancelled", {
                "step_id": step.id,
                "result": result.to_dict(),
            })
            return result

        if step.condition and not evaluate_condition(step.condition, run.results):
            result = StepResult(step.id, "skipped", f"Skipped because condition was false: {step.condition}")
            run.results[step.id] = result
            self.emit(progress_callback, "workflow", f"Skipped step {step.id}: {step.condition}")
            self.log_full_event("step_skipped", {
                "step_id": step.id,
                "condition": step.condition,
                "result": result.to_dict(),
            })
            return result

        label = f"{step.id}" if loop_iteration is None else f"{step.id} #{loop_iteration}"
        self.emit(progress_callback, "workflow", f"Running step: {label}")

        prompt = self.build_step_prompt(step, run, task)
        selected_model = self.model_for_role(step.role)
        self.log_full_event("step_started", {
            "step_id": step.id,
            "step_type": step.type,
            "role": step.role,
            "selected_model": selected_model,
            "loop_iteration": loop_iteration,
            "max_iterations": step.max_iterations,
            "prompt": prompt,
        })

        if step.type == "agentic_task":
            step_progress_callback = self.step_progress_callback(
                progress_callback,
                step,
                loop_iteration,
            )
            raw = self.agent.run_agentic_task(
                prompt,
                progress_callback=step_progress_callback,
                conversation_context=conversation_context,
                task_images=task_images,
                selected_model=selected_model,
                require_structured_result=True,
                max_iterations=step.max_iterations,
                run_log_id=self._current_debug_run_id,
                cancel_checker=cancel_checker,
            )
        else:
            raw = f"Unsupported workflow step type: {step.type}"

        result = parse_step_result(step.id, raw)
        run.results[step.id] = result
        run.timeline.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "step_id": step.id,
            "status": result.status,
            "summary": result.summary,
        })
        self.emit(progress_callback, "workflow", f"Step {step.id}: {result.status} - {result.summary[:160]}")
        self.log_full_event("step_finished", {
            "step_id": step.id,
            "step_type": step.type,
            "role": step.role,
            "selected_model": selected_model,
            "loop_iteration": loop_iteration,
            "raw_output": raw,
            "result": result.to_dict(),
        })
        return result

    def cancelled(
        self,
        cancel_checker: Callable[[], bool] | None,
        progress_callback: WorkflowProgress,
        run: WorkflowRun,
    ) -> bool:
        if not cancel_checker or not cancel_checker():
            return False

        run.status = "cancelled"
        self.emit(progress_callback, "workflow", "Workflow abort requested")
        self.log_full_event("workflow_cancelled", {
            "workflow_id": run.workflow_id,
            "results": {
                step_id: result.to_dict()
                for step_id, result in run.results.items()
            },
        })
        return True

    def step_progress_callback(
        self,
        progress_callback: WorkflowProgress,
        step: WorkflowStep,
        loop_iteration: int | None = None,
    ) -> WorkflowProgress:
        if progress_callback is None:
            return None

        label = step.id if loop_iteration is None else f"{step.id} #{loop_iteration}"

        def callback(event: Dict[str, object]) -> None:
            if isinstance(event, dict):
                enriched = dict(event)
                enriched.setdefault("workflow_step", step.id)
                enriched.setdefault("workflow_step_label", label)
                enriched.setdefault("workflow_loop_iteration", loop_iteration)
                progress_callback(enriched)
            else:
                progress_callback({
                    "kind": "progress",
                    "text": str(event),
                    "workflow_step": step.id,
                    "workflow_step_label": label,
                    "workflow_loop_iteration": loop_iteration,
                })

        return callback

    def build_step_prompt(self, step: WorkflowStep, run: WorkflowRun, task: str) -> str:
        results = {
            step_id: compact_step_result_for_context(result)
            for step_id, result in run.results.items()
        }

        return f"""
Workflow step: {step.id}
Step type: {step.type}
Step instruction:
{step.prompt}

Tool guidance:
All normal agent tools are available in every workflow step. For this step, especially consider:
{step.tool_guidance or "Use whichever tools are needed to complete the step properly."}

Iteration budget for this step:
{step.max_iterations} model/tool iterations before a forced structured finalizer.

Original user task:
{task}

Previous workflow results:
{json.dumps(results, indent=2, ensure_ascii=False)}

Use tools as needed. While using tools, always respond with the normal agent action object.

When this workflow step is complete, return one final action object using action "finish".
The finish.message value must be this workflow result object:
{{
  "summary": "short final action summary",
  "action": "finish",
  "message": {{
    "status": "passed|failed|needs_changes|blocked",
    "summary": "short workflow step result summary",
    "findings": ["specific issue or note"],
    "artifacts": ["created or inspected file names"]
  }}
}}

Important:
- Use tools when needed. Do not claim files or browser state are inaccessible before trying relevant tools.
- A step is not complete until it has enough evidence for its status.
- Do not return the workflow result object directly unless you cannot follow the finish action format.
- Do not use action "finish" until message contains the workflow result object above.
""".strip()

    def model_for_role(self, role: str) -> str:
        if role == "default":
            return self.agent.model
        return self.agent.routing_roles.get(role) or self.agent.model

    def final_summary(self, workflow: WorkflowDefinition, run: WorkflowRun) -> str:
        lines = [
            f"Workflow `{workflow.name}` finished with status `{run.status}`.",
            "",
            "Step results:",
        ]

        for result in run.results.values():
            lines.append(f"- `{result.step_id}`: {result.status} - {result.summary}")

        return "\n".join(lines)

    def emit(self, progress_callback: WorkflowProgress, kind: str, text: str) -> None:
        event = {
            "kind": kind,
            "text": text,
        }
        if progress_callback:
            progress_callback(event)

    def log_full_event(self, event_type: str, payload: Dict[str, object]) -> None:
        self.agent.append_run_log(
            "workflow",
            event_type,
            {
                "run_id": self._current_debug_run_id,
                **payload,
            },
            level="full",
            run_id=self._current_debug_run_id,
        )


def append_workflow_minimal_log(
    agent,
    task: str,
    workflow: WorkflowDefinition,
    run: WorkflowRun,
    summary: str,
    run_id: str | None = None,
) -> None:
    agent.append_run_log(
        "workflow",
        "workflow_summary",
        {
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "input": task,
            "status": run.status,
            "results": {
                step_id: compact_step_result_for_context(result)
                for step_id, result in run.results.items()
            },
            "summary": summary,
        },
        level="minimal",
        run_id=run_id,
    )


def append_workflow_debug_log(agent, task: str, workflow: WorkflowDefinition, summary: str) -> None:
    agent.append_run_log(
        "workflow",
        "workflow_summary",
        {
            "workflow_id": workflow.id,
            "task_preview": task[:500],
            "summary": summary,
        },
        level="minimal",
    )
