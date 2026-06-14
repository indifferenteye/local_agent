#!/usr/bin/env python3

import json
import logging
import os
import queue
import threading
from datetime import datetime
from typing import Any, Dict, List

from flask import Flask, Response, jsonify, render_template_string, request

from agent_core_fixed_import import OllamaAgent


app = Flask(__name__)

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

agent = OllamaAgent()
agent_lock = threading.Lock()

SESSION_FILE = os.path.join(agent.working_dir, ".agent_sessions.json")
SETTINGS_FILE = os.path.join(agent.working_dir, ".agent_settings.json")

RECENT_MESSAGES_TO_KEEP = int(os.getenv("AGENT_RECENT_MESSAGES_TO_KEEP", "30"))
SUMMARIZE_AFTER_MESSAGES = int(os.getenv("AGENT_SUMMARIZE_AFTER_MESSAGES", "60"))

messages: List[Dict[str, Any]] = []
subscribers: List[queue.Queue] = []
running = False


def load_settings() -> None:
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            selected_model = data.get("model")
            if isinstance(selected_model, str) and selected_model.strip():
                agent.model = selected_model.strip()
                print(f"Loaded selected model: {agent.model}")

    except Exception as exc:
        print(f"Could not load settings file: {exc}")


def save_settings() -> None:
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)

        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": agent.model,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    except Exception as exc:
        print(f"Could not save settings file: {exc}")


def load_messages() -> None:
    global messages

    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                messages = data
                print(f"Loaded {len(messages)} saved messages.")
    except Exception as exc:
        print(f"Could not load session file: {exc}")


def save_messages() -> None:
    try:
        os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)

        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"Could not save session file: {exc}")


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


def maybe_summarize_and_trim_memory() -> None:
    global messages

    if len(messages) <= SUMMARIZE_AFTER_MESSAGES:
        return

    old_messages = messages[:-RECENT_MESSAGES_TO_KEEP]
    recent_messages = messages[-RECENT_MESSAGES_TO_KEEP:]

    memory_candidates = []
    for msg in old_messages:
        compact = compact_message_for_memory(msg)
        if compact:
            memory_candidates.append(compact)

    if not memory_candidates:
        messages = recent_messages
        save_messages()
        return

    try:
        existing_summary = agent.load_memory_summary()
        updated_summary = agent.summarize_conversation_memory(existing_summary, memory_candidates)
        agent.save_memory_summary(updated_summary)

        messages = recent_messages
        save_messages()

        print(
            f"Summarized {len(memory_candidates)} messages. "
            f"Kept {len(messages)} recent messages."
        )

    except Exception as exc:
        print(f"Memory summarization failed: {exc}")


load_messages()
load_settings()


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Local Ollama Agent</title>
    <style>
        * {
            box-sizing: border-box;
        }

        :root {
            color-scheme: dark;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        html,
        body {
            margin: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: #101418;
            color: #f3f7fb;
        }

        body {
            display: grid;
            grid-template-rows: auto auto 1fr auto;
        }

        header {
            padding: 14px 16px;
            background: #18212b;
            border-bottom: 1px solid #263241;
            font-weight: 700;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            min-height: 52px;
        }

        .header-actions {
            display: flex;
            gap: 8px;
            align-items: center;
            flex-shrink: 0;
        }

        .small-button {
            border: 1px solid #314153;
            border-radius: 10px;
            padding: 7px 10px;
            font: inherit;
            font-size: 12px;
            background: #101820;
            color: #d7e2ee;
            cursor: pointer;
        }

        .small-button:hover {
            background: #1d2935;
        }

        .model-label {
            font-size: 12px;
            color: #9fb0c2;
            font-weight: 500;
        }

        .model-select {
            border: 1px solid #314153;
            border-radius: 10px;
            padding: 7px 10px;
            font: inherit;
            font-size: 12px;
            background: #101820;
            color: #d7e2ee;
            max-width: 260px;
        }

        .model-select:disabled {
            opacity: 0.5;
        }

        .meta {
            font-size: 12px;
            color: #9fb0c2;
            padding: 8px 16px 10px;
            background: #18212b;
            border-bottom: 1px solid #263241;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        #chat {
            min-height: 0;
            overflow-y: auto;
            padding: 16px;
            display: block;
        }

        .row {
            width: 100%;
            display: flex;
            margin-bottom: 12px;
            clear: both;
        }

        .row.user-row {
            justify-content: flex-end;
        }

        .row.agent-row,
        .row.progress-row {
            justify-content: flex-start;
        }

        .msg {
            max-width: min(920px, 92vw);
            padding: 12px 14px;
            border-radius: 14px;
            line-height: 1.45;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        .user {
            background: #255d3d;
        }

        .agent {
            background: #1f2a36;
            border: 1px solid #314153;
        }

        .progress-group {
            width: min(920px, 92vw);
            min-width: min(420px, 92vw);
            border-radius: 14px;
            background: #151d26;
            border: 1px solid #2b3a4b;
            color: #c8d6e5;
            overflow: hidden;
            display: block;
        }

        .progress-group > summary {
            cursor: pointer;
            padding: 10px 14px;
            user-select: none;
            font-size: 14px;
            min-height: 40px;
            display: block;
            overflow-wrap: anywhere;
        }

        .progress-content {
            border-top: 1px solid #2b3a4b;
            padding: 10px 12px 12px;
        }

        .step {
            margin: 8px 0;
            border: 1px solid #283747;
            border-radius: 10px;
            background: #101820;
            overflow: hidden;
            display: block;
        }

        .step > summary {
            cursor: pointer;
            padding: 9px 10px;
            font-size: 13px;
            color: #d7e2ee;
            min-height: 34px;
            display: block;
            overflow-wrap: anywhere;
        }

        .step-body {
            border-top: 1px solid #283747;
            padding: 10px;
        }

        .label {
            color: #9fb0c2;
            font-size: 12px;
            margin: 8px 0 4px;
        }

        pre {
            margin: 0;
            padding: 10px;
            background: #080d12;
            border: 1px solid #202b38;
            border-radius: 8px;
            color: #d7e2ee;
            overflow: auto;
            max-height: 280px;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            word-break: break-word;
            font-size: 12px;
        }

        .raw-model-output {
            max-height: 360px;
        }

        form {
            display: flex;
            gap: 8px;
            padding: 12px;
            background: #18212b;
            border-top: 1px solid #263241;
            min-height: 72px;
        }

        textarea {
            flex: 1;
            resize: none;
            min-height: 46px;
            max-height: 140px;
            border-radius: 12px;
            border: 1px solid #314153;
            background: #0f151b;
            color: #f3f7fb;
            padding: 10px;
            font: inherit;
            min-width: 0;
        }

        button {
            border: 0;
            border-radius: 12px;
            padding: 0 18px;
            font: inherit;
            font-weight: 700;
            background: #32a852;
            color: white;
            cursor: pointer;
            flex-shrink: 0;
        }

        button:disabled {
            opacity: 0.5;
            cursor: default;
        }

        @media (max-width: 760px) {
            header {
                align-items: flex-start;
                flex-direction: column;
            }

            .header-actions {
                width: 100%;
                flex-wrap: wrap;
            }

            .model-select {
                flex: 1;
                min-width: 180px;
                max-width: none;
            }
        }

        @media (max-width: 640px) {
            #chat {
                padding: 10px;
            }

            .msg,
            .progress-group {
                max-width: 96vw;
                width: 96vw;
            }

            form {
                padding: 10px;
            }

            button {
                padding: 0 12px;
            }
        }
    </style>
</head>
<body>
    <header>
        <div>Local Ollama Agent</div>
        <div class="header-actions">
            <label class="model-label" for="modelSelect">Model</label>
            <select class="model-select" id="modelSelect"></select>
            <button class="small-button" id="clear">Clear chat</button>
        </div>
    </header>

    <div class="meta" id="meta">
        Model: {{ model }} | Workdir: {{ workdir }}
    </div>

    <main id="chat"></main>

    <form id="form">
        <textarea id="task" placeholder="Type a task..." autofocus></textarea>
        <button id="send" type="submit">Send</button>
    </form>

    <script>
        const chat = document.getElementById("chat");
        const form = document.getElementById("form");
        const taskInput = document.getElementById("task");
        const sendButton = document.getElementById("send");
        const clearButton = document.getElementById("clear");
        const modelSelect = document.getElementById("modelSelect");
        const meta = document.getElementById("meta");

        let lastRenderedSignature = "";
        let renderTimer = null;

        function scrollToBottomSoon() {
            requestAnimationFrame(() => {
                chat.scrollTop = chat.scrollHeight;
                requestAnimationFrame(() => {
                    chat.scrollTop = chat.scrollHeight;
                });
            });
        }

        function makeRow(kind) {
            const row = document.createElement("div");
            row.className = "row " + kind + "-row";
            return row;
        }

        function addMessageElement(role, text) {
            const row = makeRow(role === "user" ? "user" : "agent");
            const div = document.createElement("div");
            div.className = "msg " + role;
            div.textContent = text;
            row.appendChild(div);
            chat.appendChild(row);
        }

        function makePre(text, extraClass) {
            const pre = document.createElement("pre");
            if (extraClass) pre.className = extraClass;
            pre.textContent = text || "";
            return pre;
        }

        function addProgressGroup(progressItems) {
            const row = makeRow("progress");

            const details = document.createElement("details");
            details.className = "progress-group";
            details.open = false;

            const summary = document.createElement("summary");

            const latest = progressItems.length
                ? progressItems[progressItems.length - 1].text
                : "No progress yet";

            const latestShort = latest && latest.length > 80
                ? latest.slice(0, 80) + "..."
                : latest;

            summary.textContent =
                "Progress details (" + progressItems.length + ") · latest: " + latestShort;

            const content = document.createElement("div");
            content.className = "progress-content";

            const steps = new Map();

            for (const item of progressItems) {
                const iteration = item.iteration || "general";

                if (!steps.has(iteration)) {
                    steps.set(iteration, []);
                }

                steps.get(iteration).push(item);
            }

            for (const [iteration, items] of steps.entries()) {
                const step = document.createElement("details");
                step.className = "step";
                step.open = false;

                const stepSummary = document.createElement("summary");

                const summaryItem =
                    items.find(i => i.kind === "summary") ||
                    items.find(i => i.kind === "status") ||
                    items[items.length - 1];

                const action = summaryItem.action ? " · " + summaryItem.action : "";
                stepSummary.textContent =
                    iteration === "general"
                        ? "General progress"
                        : "Step " + iteration + action + ": " + summaryItem.text;

                const body = document.createElement("div");
                body.className = "step-body";

                for (const item of items) {
                    const label = document.createElement("div");
                    label.className = "label";
                    label.textContent = item.kind || "progress";
                    body.appendChild(label);

                    const isRaw = item.kind === "model_output";
                    body.appendChild(makePre(item.text || "", isRaw ? "raw-model-output" : ""));
                }

                step.appendChild(stepSummary);
                step.appendChild(body);
                content.appendChild(step);
            }

            details.appendChild(summary);
            details.appendChild(content);
            row.appendChild(details);
            chat.appendChild(row);
        }

        function getRenderSignature(messages, running) {
            if (!messages || messages.length === 0) {
                return "empty:" + running;
            }

            const last = messages[messages.length - 1];
            return messages.length + ":" + running + ":" + last.role + ":" + last.timestamp + ":" + String(last.text || "").length;
        }

        function renderMessages(messages, running) {
            const signature = getRenderSignature(messages, running);

            if (signature === lastRenderedSignature) {
                return;
            }

            const wasNearBottom =
                chat.scrollTop + chat.clientHeight >= chat.scrollHeight - 120;

            chat.innerHTML = "";

            let pendingProgress = [];

            for (const msg of messages) {
                if (msg.role === "progress") {
                    pendingProgress.push(msg);
                    continue;
                }

                if (pendingProgress.length > 0) {
                    addProgressGroup(pendingProgress);
                    pendingProgress = [];
                }

                if (msg.role === "status") {
                    continue;
                }

                addMessageElement(msg.role, msg.text || "");
            }

            if (pendingProgress.length > 0) {
                addProgressGroup(pendingProgress);
            }

            lastRenderedSignature = signature;

            if (running || wasNearBottom) {
                scrollToBottomSoon();
            }

            chat.style.display = "none";
            chat.offsetHeight;
            chat.style.display = "block";

            if (running || wasNearBottom) {
                scrollToBottomSoon();
            }
        }

        async function loadModels() {
            try {
                const res = await fetch("/models", { cache: "no-store" });
                const data = await res.json();

                modelSelect.innerHTML = "";

                for (const model of data.models || []) {
                    const option = document.createElement("option");
                    option.value = model;
                    option.textContent = model;

                    if (model === data.current_model) {
                        option.selected = true;
                    }

                    modelSelect.appendChild(option);
                }

                if ((data.models || []).length === 0) {
                    const option = document.createElement("option");
                    option.value = "";
                    option.textContent = "No models found";
                    modelSelect.appendChild(option);
                    modelSelect.disabled = true;
                }

                meta.textContent = "Model: " + data.current_model + " | Workdir: " + data.workdir;

            } catch (err) {
                console.error("Could not load models:", err);
            }
        }

        async function loadMessages() {
            try {
                const res = await fetch("/messages", { cache: "no-store" });
                const data = await res.json();

                renderMessages(data.messages || [], data.running === true);

                sendButton.disabled = data.running === true;
                modelSelect.disabled = data.running === true;

                if (data.model) {
                    meta.textContent = "Model: " + data.model + " | Workdir: " + data.workdir;
                }
            } catch (err) {
                console.error("Could not load messages:", err);
            }
        }

        function scheduleLoadMessages() {
            if (renderTimer) return;

            renderTimer = setTimeout(async () => {
                renderTimer = null;
                await loadMessages();
            }, 100);
        }

        setInterval(loadMessages, 1000);

        try {
            const events = new EventSource("/events");

            events.onmessage = () => {
                scheduleLoadMessages();
            };

            events.onerror = () => {
                console.log("EventSource disconnected or retrying...");
            };
        } catch (err) {
            console.log("EventSource unavailable:", err);
        }

        form.addEventListener("submit", async (e) => {
            e.preventDefault();

            const text = taskInput.value.trim();
            if (!text) return;

            taskInput.value = "";
            sendButton.disabled = true;

            try {
                const res = await fetch("/task", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ task: text })
                });

                if (!res.ok) {
                    const err = await res.text();
                    console.error(err);
                }

                await loadMessages();
            } catch (err) {
                console.error("Submit failed:", err);
            }
        });

        modelSelect.addEventListener("change", async () => {
            const model = modelSelect.value;
            if (!model) return;

            try {
                const res = await fetch("/model", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ model })
                });

                if (!res.ok) {
                    const err = await res.text();
                    console.error(err);
                    await loadModels();
                    return;
                }

                await loadModels();
                await loadMessages();

            } catch (err) {
                console.error("Could not change model:", err);
                await loadModels();
            }
        });

        clearButton.addEventListener("click", async () => {
            if (!confirm("Clear the saved chat history? This will not clear the long-term memory summary.")) return;

            try {
                await fetch("/clear", { method: "POST" });
                lastRenderedSignature = "";
                await loadMessages();
            } catch (err) {
                console.error("Clear failed:", err);
            }
        });

        taskInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                form.requestSubmit();
            }
        });

        window.addEventListener("resize", () => {
            lastRenderedSignature = "";
            loadMessages();
        });

        loadModels();
        loadMessages();
    </script>
</body>
</html>
"""


def normalize_progress_event(event: Any) -> Dict[str, Any]:
    if isinstance(event, dict):
        return {
            "role": "progress",
            "kind": str(event.get("kind", "progress")),
            "iteration": event.get("iteration"),
            "action": event.get("action"),
            "text": str(event.get("text", "")),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    return {
        "role": "progress",
        "kind": "progress",
        "iteration": None,
        "action": None,
        "text": str(event),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def broadcast_message(msg: Dict[str, Any]) -> None:
    msg.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))

    messages.append(msg)
    save_messages()

    dead = []
    for q in subscribers:
        try:
            q.put_nowait(msg)
        except Exception:
            dead.append(q)

    for q in dead:
        if q in subscribers:
            subscribers.remove(q)


def broadcast(role: str, text: str) -> None:
    broadcast_message({
        "role": role,
        "text": text,
    })


@app.route("/")
def index():
    return render_template_string(
        HTML,
        model=agent.model,
        workdir=agent.working_dir,
    )


@app.route("/messages")
def get_messages():
    return jsonify({
        "messages": messages,
        "running": running,
        "model": agent.model,
        "workdir": agent.working_dir,
    })


@app.route("/models")
def get_models():
    models = agent.list_installed_models()

    return jsonify({
        "models": models,
        "current_model": agent.model,
        "workdir": agent.working_dir,
    })


@app.route("/model", methods=["POST"])
def set_model():
    global running

    if running:
        return jsonify({
            "error": "Cannot change model while a task is running",
        }), 409

    data = request.get_json(force=True)
    model = str(data.get("model", "")).strip()

    if not model:
        return jsonify({
            "error": "Missing model",
        }), 400

    installed_models = agent.list_installed_models()

    if model not in installed_models:
        return jsonify({
            "error": f"Model is not installed: {model}",
            "installed_models": installed_models,
        }), 400

    agent.model = model
    save_settings()

    broadcast("status", f"Model changed to {model}")

    return jsonify({
        "model": agent.model,
    })


@app.route("/events")
def events():
    q = queue.Queue()
    subscribers.append(q)

    def stream():
        try:
            while True:
                msg = q.get()
                yield f"data: {json.dumps(msg)}\\n\\n"
        except GeneratorExit:
            if q in subscribers:
                subscribers.remove(q)

    response = Response(stream(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/task", methods=["POST"])
def run_task():
    global running

    data = request.get_json(force=True)
    task = data.get("task", "").strip()

    if not task:
        return jsonify({"error": "Missing task"}), 400

    if running:
        return jsonify({"error": "Task already running"}), 409

    broadcast("user", task)
    running = True

    def worker():
        global running

        try:
            with agent_lock:
                def progress(event: Any) -> None:
                    broadcast_message(normalize_progress_event(event))

                final = agent.run_agentic_task(task, progress_callback=progress)
                broadcast("agent", final)

                maybe_summarize_and_trim_memory()

        finally:
            running = False
            broadcast("status", "idle")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    return jsonify({"started": True})


@app.route("/clear", methods=["POST"])
def clear_messages():
    global messages

    messages = []
    save_messages()
    broadcast("status", "cleared")

    return jsonify({"cleared": True})


@app.route("/memory")
def get_memory():
    return jsonify({
        "summary": agent.load_memory_summary(),
    })


@app.route("/clear-memory", methods=["POST"])
def clear_memory():
    agent.save_memory_summary("")
    return jsonify({"cleared": True})


if __name__ == "__main__":
    print("Testing Ollama connection...")

    if not agent.wait_for_ollama():
        print("Warning: Ollama is not reachable yet.")

    print(f"Session file: {SESSION_FILE}")
    print(f"Settings file: {SETTINGS_FILE}")
    print(f"Memory summary file: {agent.memory_summary_file}")
    print("Starting web app on http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, threaded=True)