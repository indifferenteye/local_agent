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

# Hide noisy Flask request logs like: GET /messages 200
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

agent = OllamaAgent()
agent_lock = threading.Lock()

SESSION_FILE = os.path.join(agent.working_dir, ".agent_sessions.json")

messages: List[Dict[str, Any]] = []
subscribers: List[queue.Queue] = []
running = False


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


load_messages()


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Local Ollama Agent</title>
    <style>
        :root {
            color-scheme: dark;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        body {
            margin: 0;
            background: #101418;
            color: #f3f7fb;
            height: 100vh;
            display: flex;
            flex-direction: column;
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
        }

        .header-actions {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .small-button {
            border: 1px solid #314153;
            border-radius: 10px;
            padding: 7px 10px;
            font: inherit;
            font-size: 12px;
            background: #101820;
            color: #d7e2ee;
        }

        .small-button:hover {
            background: #1d2935;
        }

        .meta {
            font-size: 12px;
            color: #9fb0c2;
            padding: 0 16px 10px;
            background: #18212b;
            border-bottom: 1px solid #263241;
        }

        #chat {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .msg {
            max-width: 920px;
            padding: 12px 14px;
            border-radius: 14px;
            line-height: 1.45;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .user {
            align-self: flex-end;
            background: #255d3d;
        }

        .agent {
            align-self: flex-start;
            background: #1f2a36;
            border: 1px solid #314153;
        }

        .progress-group {
            align-self: flex-start;
            max-width: 920px;
            width: min(920px, 100%);
            border-radius: 14px;
            background: #151d26;
            border: 1px solid #2b3a4b;
            color: #c8d6e5;
            overflow: hidden;
        }

        .progress-group > summary {
            cursor: pointer;
            padding: 10px 14px;
            user-select: none;
            font-size: 14px;
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
        }

        .step > summary {
            cursor: pointer;
            padding: 9px 10px;
            font-size: 13px;
            color: #d7e2ee;
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
        }

        button {
            border: 0;
            border-radius: 12px;
            padding: 0 18px;
            font: inherit;
            font-weight: 700;
            background: #32a852;
            color: white;
        }

        button:disabled {
            opacity: 0.5;
        }
    </style>
</head>
<body>
    <header>
        <div>Local Ollama Agent</div>
        <div class="header-actions">
            <button class="small-button" id="clear">Clear chat</button>
        </div>
    </header>

    <div class="meta">
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

        let lastRenderedCount = -1;

        function scrollToBottom() {
            chat.scrollTop = chat.scrollHeight;
        }

        function addMessageElement(role, text) {
            const div = document.createElement("div");
            div.className = "msg " + role;
            div.textContent = text;
            chat.appendChild(div);
        }

        function makePre(text, extraClass) {
            const pre = document.createElement("pre");
            if (extraClass) pre.className = extraClass;
            pre.textContent = text || "";
            return pre;
        }

        function addProgressGroup(progressItems) {
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
            chat.appendChild(details);
        }

        function renderMessages(messages) {
            if (messages.length === lastRenderedCount) {
                return;
            }

            const wasNearBottom =
                chat.scrollTop + chat.clientHeight >= chat.scrollHeight - 80;

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

                addMessageElement(msg.role, msg.text);
            }

            if (pendingProgress.length > 0) {
                addProgressGroup(pendingProgress);
            }

            lastRenderedCount = messages.length;

            if (wasNearBottom) {
                scrollToBottom();
            }
        }

        async function loadMessages() {
            try {
                const res = await fetch("/messages", { cache: "no-store" });
                const data = await res.json();
                renderMessages(data.messages);
                sendButton.disabled = data.running === true;
            } catch (err) {
                console.error("Could not load messages:", err);
            }
        }

        setInterval(loadMessages, 1000);

        try {
            const events = new EventSource("/events");

            events.onmessage = () => {
                loadMessages();
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

        clearButton.addEventListener("click", async () => {
            if (!confirm("Clear the saved chat history?")) return;

            try {
                await fetch("/clear", { method: "POST" });
                lastRenderedCount = -1;
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


if __name__ == "__main__":
    print("Testing Ollama connection...")

    if not agent.wait_for_ollama():
        print("Warning: Ollama is not reachable yet.")

    print(f"Session file: {SESSION_FILE}")
    print("Starting web app on http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, threaded=True)