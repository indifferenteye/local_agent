# Agent Guide

## App Purpose

This repository contains `chatgt`, a local web UI for running an Ollama-backed agent against a mounted work directory. The app is designed for local-first coding, debugging, browser/UI inspection, image-aware tasks when a local vision model is configured, and reusable workflows that can plan, implement, test, and iterate.

The default interaction is a normal agent task. The workflow selector runs a reusable sequence instead of a single free-form task.

## Key Modules

- `chatgt/agent_core.py`: Ollama calls, model selection, tool loop, routing integration, browser/file tools, and run logging.
- `chatgt/agent_routing.py`: deterministic task classification, optional router-model classification, and policy-based role selection.
- `chatgt/agent_workflows.py`: workflow definitions, step execution, conditions, loops, and workflow summaries.
- `chatgt/routes.py`: Flask routes, SSE progress events, upload handling, workflow entry points, and settings endpoints.
- `chatgt/templates/index.html`: the single-page UI.
- `chatgt/persistence.py`: durable chat/settings persistence.
- `chatgt/app_state.py`: process-wide app state and settings mutation helpers.

## Routing Model

Routing is policy-driven. Deterministic rules classify obvious tasks first, and the router model is only used when the deterministic result is uncertain. The classifier does not directly choose the execution model; the routing policy chooses from configured roles.

Configured roles are:

- default: the selected top-level model.
- router: lightweight classification and cheap simple tasks.
- planner: planning, architecture, debugging, and non-trivial reasoning.
- coding: code, file edits, tests, browser UI work, and tool-heavy tasks.
- vision: uploaded-image analysis or visual edit requests.

Unset roles fall back to the default selected model. Quality modes are `economy`, `balanced`, and `high_quality`; code/file/tool-risk tasks should not be routed below the coding role when it is configured.

## Workflow Model

Workflows are reusable agent execution patterns. Built-in workflows currently cover code tasks, frontend visual work, and debugging. A workflow can contain steps and loops with conditions such as `check.status == "passed"` or `check.findings_count > 0`.

All workflow steps should have access to the full tool set. Step role and `tool_guidance` should encourage the right behavior, but should not disable tools. Verification steps should be agentic, inspect the workspace/UI as needed, and return structured JSON:

```json
{"status":"passed","summary":"...","findings":[],"artifacts":[]}
```

If a step cannot produce structured JSON, the workflow treats it as `needs_changes` so loops can repair or escalate instead of silently passing.

## Logging

There are two different kinds of app state:

- `.agent_sessions.json`: durable chat transcript. By default this stores only user messages and final agent messages.
- `.agent_runs.jsonl`: compact routing/workflow run index for diagnostics.

Run log levels:

- `off`: no diagnostic run log.
- `minimal`: input/output summaries for routing and workflows.
- `full`: compact index events plus complete internal details.

In full mode, large internal fields such as workflow prompts and raw model outputs are written under `.agent_run_details/<run_id>/...`. The JSONL index keeps previews and `details_ref` paths so it remains readable and easy to parse.

Do not use durable chat persistence for progress spam. Keep live progress/status messages transient unless a debugging setting explicitly enables persistence.

## Settings

Settings are loaded and saved through `.agent_settings.json` via `load_settings()` and `save_settings()`. The settings modal controls context budget, routing enablement, routing quality, run log level, and model roles. The top-level model selector remains the default/fallback model.

## Verification

Use these checks after changing routing, workflows, persistence, or UI behavior:

```powershell
python -B -m unittest discover
python -B -m py_compile chatgt\agent_core.py chatgt\agent_routing.py chatgt\agent_workflows.py chatgt\routes.py chatgt\app_state.py chatgt\persistence.py
```

If Python creates `__pycache__` directories during verification, clean only generated cache directories after confirming the paths are inside this repository.

For browser/UI changes, start the app or use the existing running server, then manually smoke test the settings modal, normal tasks, workflow tasks, and image-upload path when relevant.

## Development Notes

Prefer small changes that preserve the current agent loop. Keep routing and workflow decisions debuggable through progress events and run logs, but keep saved chat compact. When adding new workflow behavior, add focused `unittest` coverage instead of introducing a new test dependency.
