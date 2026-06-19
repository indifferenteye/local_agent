#!/usr/bin/env python3

import json
import queue
import threading
from typing import Any

from flask import Response, jsonify, render_template, request, send_file

import app_state as state
from chat_images import image_message_data, safe_image_path, save_uploaded_image
from commands import handle_slash_command
from events import broadcast, broadcast_message, normalize_progress_event
from memory import maybe_summarize_memory, prepare_context_for_next_task
from persistence import load_memory_state, save_memory_state, save_messages, save_settings
from agent_workflows import (
    WorkflowRunner,
    built_in_workflow_ids,
    clone_workflow,
    delete_custom_workflow,
    get_workflow,
    get_workflow_detail,
    save_custom_workflow,
    workflow_from_dict,
    workflow_options_for_workdir,
    workflow_to_dict,
    workflow_validation_errors,
)


def register_routes(app) -> None:
    def start_agent_task(
        task: str,
        task_images: list[str] | None = None,
        display_task: str | None = None,
        workflow_id: str | None = None,
    ):
        broadcast(
            "user",
            display_task if display_task is not None else task,
            images=[image_message_data(path) for path in task_images or []],
        )
        state.running = True
        cancel_event = threading.Event()
        state.cancel_event = cancel_event

        def worker():
            try:
                with state.agent_lock:
                    def is_cancelled() -> bool:
                        return cancel_event.is_set()

                    def progress(event: Any) -> None:
                        broadcast_message(normalize_progress_event(event))

                    context_bundle = prepare_context_for_next_task()
                    conversation_context = context_bundle["recent_context"]
                    context_status = context_bundle["status"]

                    broadcast_message(normalize_progress_event({
                        "kind": "status",
                        "text": (
                            "Prepared context: "
                            f"{context_status['recent_messages']} recent messages, "
                            f"{context_status['memory_summary_chars']} summary chars"
                        ),
                    }))

                    if workflow_id:
                        workflow = get_workflow(workflow_id, state.agent.working_dir)
                        workflow_detail = get_workflow_detail(workflow_id, state.agent.working_dir)
                        if workflow is None:
                            final = f"Unknown workflow: {workflow_id}"
                        else:
                            source = (workflow_detail or {}).get("source", "unknown")
                            broadcast_message(normalize_progress_event({
                                "kind": "status",
                                "text": f"Workflow selected: {workflow.name} ({source})",
                            }))
                            runner = WorkflowRunner(state.agent)
                            final = runner.run(
                                workflow,
                                task,
                                progress_callback=progress,
                                conversation_context=conversation_context,
                                task_images=task_images or [],
                                cancel_checker=is_cancelled,
                            )
                    else:
                        routing_decision = state.agent.route_current_task(
                            task,
                            conversation_context=conversation_context,
                            task_images=task_images or [],
                        )
                        state.agent.log_routing_decision(task, routing_decision)
                        classification = routing_decision.classification
                        classification_type = (
                            classification.task_type if classification else "unknown"
                        )
                        fallback = " via default fallback" if routing_decision.fallback_used else ""

                        broadcast_message(normalize_progress_event({
                            "kind": "status",
                            "text": (
                                "Routing: "
                                f"{classification_type} -> {routing_decision.selected_role} "
                                f"({routing_decision.selected_model}){fallback}. "
                                f"{routing_decision.reason}"
                            ),
                        }))

                        final = state.agent.run_agentic_task(
                            task,
                            progress_callback=progress,
                            conversation_context=conversation_context,
                            task_images=task_images or [],
                            selected_model=routing_decision.selected_model,
                            routing_decision=routing_decision,
                            cancel_checker=is_cancelled,
                        )
                    if is_cancelled():
                        final = "Task aborted."

                    broadcast("agent", final, images=state.agent.consume_output_images())

                    if not is_cancelled():
                        maybe_summarize_memory()

            finally:
                state.running = False
                if state.cancel_event is cancel_event:
                    state.cancel_event = None
                broadcast("status", "idle")

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            model=state.agent.model,
            think_mode=state.agent.get_think_mode_label(),
            workdir=state.agent.working_dir,
        )

    @app.route("/messages")
    def get_messages():
        return jsonify({
            "messages": state.messages,
            "running": state.running,
            "model": state.agent.model,
            "think_mode": state.agent.get_think_mode_label(),
            "workdir": state.agent.working_dir,
            "context": context_payload(),
        })

    @app.route("/models")
    def get_models():
        models = state.agent.list_installed_models()

        return jsonify({
            "models": models,
            "current_model": state.agent.model,
            "think_mode": state.agent.get_think_mode_label(),
            "workdir": state.agent.working_dir,
            "context": context_payload(),
        })

    @app.route("/workflows")
    def get_workflows():
        return jsonify({
            "workflows": workflow_options_for_workdir(state.agent.working_dir),
        })

    def workflow_error(message: str, status: int = 400):
        return jsonify({"error": message, "errors": [message]}), status

    def workflow_from_request():
        data = request.get_json(force=True)
        workflow_data = data.get("workflow", data) if isinstance(data, dict) else {}
        if not isinstance(workflow_data, dict):
            raise ValueError("Workflow payload must be an object")
        return workflow_from_dict(workflow_data)

    @app.route("/workflows/<workflow_id>")
    def get_workflow_api(workflow_id: str):
        detail = get_workflow_detail(workflow_id, state.agent.working_dir)
        if detail is None:
            return workflow_error(f"Unknown workflow: {workflow_id}", 404)
        return jsonify({"workflow": detail})

    @app.route("/workflows", methods=["POST"])
    def create_workflow_api():
        if state.running:
            return workflow_error("Cannot change workflows while a task is running", 409)
        try:
            workflow = workflow_from_request()
            errors = workflow_validation_errors(workflow, reserved_ids=built_in_workflow_ids())
            if errors:
                return jsonify({"error": errors[0], "errors": errors}), 400
            saved = save_custom_workflow(state.agent.working_dir, workflow)
            return jsonify({"workflow": {**workflow_to_dict(saved), "editable": True, "source": "custom"}})
        except ValueError as exc:
            return workflow_error(str(exc), 400)

    @app.route("/workflows/<workflow_id>", methods=["PUT"])
    def update_workflow_api(workflow_id: str):
        if state.running:
            return workflow_error("Cannot change workflows while a task is running", 409)
        if workflow_id in built_in_workflow_ids():
            return workflow_error(f"Cannot update built-in workflow: {workflow_id}", 400)
        try:
            workflow = workflow_from_request()
            errors = workflow_validation_errors(workflow, reserved_ids=built_in_workflow_ids())
            if errors:
                return jsonify({"error": errors[0], "errors": errors}), 400
            saved = save_custom_workflow(state.agent.working_dir, workflow, replacing_id=workflow_id)
            return jsonify({"workflow": {**workflow_to_dict(saved), "editable": True, "source": "custom"}})
        except ValueError as exc:
            return workflow_error(str(exc), 400)

    @app.route("/workflows/<workflow_id>", methods=["DELETE"])
    def delete_workflow_api(workflow_id: str):
        if state.running:
            return workflow_error("Cannot change workflows while a task is running", 409)
        if workflow_id in built_in_workflow_ids():
            return workflow_error(f"Cannot delete built-in workflow: {workflow_id}", 400)
        if not delete_custom_workflow(state.agent.working_dir, workflow_id):
            return workflow_error(f"Unknown custom workflow: {workflow_id}", 404)
        return jsonify({"deleted": True})

    @app.route("/workflows/<workflow_id>/clone", methods=["POST"])
    def clone_workflow_api(workflow_id: str):
        if state.running:
            return workflow_error("Cannot change workflows while a task is running", 409)
        data = request.get_json(silent=True) or {}
        try:
            cloned = clone_workflow(
                state.agent.working_dir,
                workflow_id,
                new_id=str(data.get("id", "") or "").strip() or None,
                new_name=str(data.get("name", "") or "").strip() or None,
            )
            return jsonify({"workflow": {**workflow_to_dict(cloned), "editable": True, "source": "custom"}})
        except ValueError as exc:
            return workflow_error(str(exc), 400)

    @app.route("/workflows/import", methods=["POST"])
    def import_workflow_api():
        if state.running:
            return workflow_error("Cannot change workflows while a task is running", 409)
        try:
            workflow = workflow_from_request()
            errors = workflow_validation_errors(workflow, reserved_ids=built_in_workflow_ids())
            if errors:
                return jsonify({"error": errors[0], "errors": errors}), 400
            saved = save_custom_workflow(state.agent.working_dir, workflow)
            return jsonify({"workflow": {**workflow_to_dict(saved), "editable": True, "source": "custom"}})
        except ValueError as exc:
            return workflow_error(str(exc), 400)

    @app.route("/workflows/<workflow_id>/export")
    def export_workflow_api(workflow_id: str):
        detail = get_workflow_detail(workflow_id, state.agent.working_dir)
        if detail is None:
            return workflow_error(f"Unknown workflow: {workflow_id}", 404)
        export_data = {
            key: value for key, value in detail.items()
            if key not in {"editable", "source"}
        }
        return Response(
            json.dumps(export_data, indent=2) + "\n",
            mimetype="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={workflow_id}.workflow.json",
            },
        )

    @app.route("/model", methods=["POST"])
    def set_model():
        if state.running:
            return jsonify({
                "error": "Cannot change model while a task is running",
            }), 409

        data = request.get_json(force=True)
        model = str(data.get("model", "")).strip()

        if not model:
            return jsonify({
                "error": "Missing model",
            }), 400

        installed_models = state.agent.list_installed_models()

        if model not in installed_models:
            return jsonify({
                "error": f"Model is not installed: {model}",
                "installed_models": installed_models,
            }), 400

        state.agent.model = model
        save_settings()

        broadcast("status", f"Model changed to {model}")

        return jsonify({
            "model": state.agent.model,
            "think_mode": state.agent.get_think_mode_label(),
        })

    def context_payload() -> dict[str, Any]:
        summary = state.agent.load_memory_summary()
        recent_context = state.context_manager.recent_conversation(state.messages)

        return {
            "recent_context_messages": state.RECENT_CONVERSATION_CONTEXT_MESSAGES,
            "recent_context_chars": state.RECENT_CONVERSATION_CONTEXT_CHARS,
            "context_target_tokens": state.CONTEXT_TARGET_TOKENS,
            "ollama_num_ctx": state.agent.ollama_num_ctx,
            "status": state.context_manager.context_status(summary, recent_context),
            "routing": {
                "enabled": state.agent.routing_enabled,
                "quality_mode": state.agent.routing_quality_mode,
                "run_log_level": state.agent.run_log_level,
                "run_log_file": state.agent.run_log_file,
                "run_log_detail_dir": state.agent.run_log_detail_dir,
                "roles": state.agent.routing_roles,
            },
        }

    @app.route("/context-settings", methods=["GET", "POST"])
    def context_settings():
        if request.method == "GET":
            return jsonify(context_payload())

        if state.running:
            return jsonify({
                "error": "Cannot change context settings while a task is running",
            }), 409

        data = request.get_json(force=True)
        state.update_context_settings(data)
        state.update_routing_settings(data)
        save_settings()

        broadcast("status", "Settings updated")
        return jsonify(context_payload())

    @app.route("/events")
    def events():
        q = queue.Queue()
        state.subscribers.append(q)

        def stream():
            try:
                while True:
                    msg = q.get()
                    yield f"data: {json.dumps(msg)}\\n\\n"
            except GeneratorExit:
                if q in state.subscribers:
                    state.subscribers.remove(q)

        response = Response(stream(), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @app.route("/workdir-image/<path:filename>")
    def workdir_image(filename: str):
        try:
            return send_file(safe_image_path(filename))
        except FileNotFoundError:
            return jsonify({"error": "Image not found"}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/upload", methods=["POST"])
    def upload_images():
        task = request.form.get("task", "").strip()
        files = request.files.getlist("images")

        if not task and not files:
            return jsonify({"error": "Missing task or image"}), 400

        if state.running:
            return jsonify({"error": "Task already running"}), 409

        images = []

        try:
            for file in files:
                if not file or not file.filename:
                    continue
                images.append(save_uploaded_image(file))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        if task:
            workflow_id = request.form.get("workflow", "").strip() or None
            image_paths = [image["filename"] for image in images]
            image_context = ""
            if image_paths:
                image_context = "\n\nUploaded image files:\n" + "\n".join(
                    f"- {path}" for path in image_paths
                )

            start_agent_task(
                task + image_context,
                image_paths,
                display_task=task,
                workflow_id=workflow_id,
            )
            return jsonify({"started": True, "images": images})

        broadcast("user", "Uploaded image.", images=images)
        return jsonify({"uploaded": True, "images": images})

    @app.route("/task", methods=["POST"])
    def run_task():
        data = request.get_json(force=True)
        task = data.get("task", "").strip()
        workflow_id = str(data.get("workflow", "") or "").strip() or None

        if not task:
            return jsonify({"error": "Missing task"}), 400

        if task.startswith("/"):
            handled = handle_slash_command(task)
            if handled:
                return jsonify({"handled": True})

        if state.running:
            return jsonify({"error": "Task already running"}), 409

        start_agent_task(task, workflow_id=workflow_id)

        return jsonify({"started": True})

    @app.route("/abort", methods=["POST"])
    def abort_task():
        if not state.running or state.cancel_event is None:
            return jsonify({"aborted": False, "running": False})

        state.cancel_event.set()
        broadcast_message(normalize_progress_event({
            "kind": "status",
            "text": "Abort requested. Stopping at the next safe checkpoint...",
        }))
        broadcast("status", "abort requested")

        return jsonify({"aborted": True, "running": True})

    @app.route("/clear", methods=["POST"])
    def clear_messages():
        state.messages.clear()
        save_messages()
        state.agent.clear_run_logs()

        save_memory_state({
            "summarized_until_index": 0,
        })

        broadcast("status", "cleared")

        return jsonify({"cleared": True})

    @app.route("/memory")
    def get_memory():
        return jsonify({
            "summary": state.agent.load_memory_summary(),
            "state": load_memory_state(),
        })

    @app.route("/clear-memory", methods=["POST"])
    def clear_memory():
        state.agent.save_memory_summary("")
        save_memory_state({
            "summarized_until_index": 0,
        })

        return jsonify({"cleared": True})
