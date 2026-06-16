#!/usr/bin/env python3

import json
import queue
import threading
from typing import Any

from flask import Response, jsonify, render_template, request

import app_state as state
from commands import handle_slash_command
from events import broadcast, broadcast_message, normalize_progress_event
from memory import get_recent_conversation_context, maybe_summarize_memory
from persistence import load_memory_state, save_memory_state, save_messages, save_settings


def register_routes(app) -> None:
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
        })

    @app.route("/models")
    def get_models():
        models = state.agent.list_installed_models()

        return jsonify({
            "models": models,
            "current_model": state.agent.model,
            "think_mode": state.agent.get_think_mode_label(),
            "workdir": state.agent.working_dir,
        })

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

    @app.route("/task", methods=["POST"])
    def run_task():
        data = request.get_json(force=True)
        task = data.get("task", "").strip()

        if not task:
            return jsonify({"error": "Missing task"}), 400

        if task.startswith("/"):
            handled = handle_slash_command(task)
            if handled:
                return jsonify({"handled": True})

        if state.running:
            return jsonify({"error": "Task already running"}), 409

        broadcast("user", task)
        state.running = True

        def worker():
            try:
                with state.agent_lock:
                    def progress(event: Any) -> None:
                        broadcast_message(normalize_progress_event(event))

                    conversation_context = get_recent_conversation_context()

                    final = state.agent.run_agentic_task(
                        task,
                        progress_callback=progress,
                        conversation_context=conversation_context,
                    )
                    broadcast("agent", final)

                    maybe_summarize_memory()

            finally:
                state.running = False
                broadcast("status", "idle")

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        return jsonify({"started": True})

    @app.route("/clear", methods=["POST"])
    def clear_messages():
        state.messages.clear()
        save_messages()

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
