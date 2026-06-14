#!/usr/bin/env python3

import json
import queue
import threading
from typing import Dict, List

from flask import Flask, Response, jsonify, render_template_string, request

from agent_core_fixed_import import OllamaAgent


app = Flask(__name__)

agent = OllamaAgent()
agent_lock = threading.Lock()

messages: List[Dict[str, str]] = []
subscribers: List[queue.Queue] = []


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
            border-radius: 14px;
            background: #151d26;
            border: 1px solid #2b3a4b;
            color: #c8d6e5;
            overflow: hidden;
        }

        .progress-group summary {
            cursor: pointer;
            padding: 10px 14px;
            user-select: none;
            font-size: 14px;
        }

        .progress-lines {
            border-top: 1px solid #2b3a4b;
            padding: 8px 14px 12px;
            font-size: 13px;
            color: #9fb0c2;
            white-space: pre-wrap;
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
    <header>Local Ollama Agent</header>
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

        function addProgressGroup(progressItems) {
            const details = document.createElement("details");
            details.className = "progress-group";
            details.open = false;

            const summary = document.createElement("summary");
            const latest = progressItems.length
                ? progressItems[progressItems.length - 1].text
                : "No progress yet";

            const latestShort = latest.length > 80 ? latest.slice(0, 80) + "..." : latest;
            summary.textContent =
                "Progress details (" + progressItems.length + ") · latest: " + latestShort;

            const lines = document.createElement("div");
            lines.className = "progress-lines";
            lines.textContent = progressItems.map(m => m.text).join("\\n");

            details.appendChild(summary);
            details.appendChild(lines);
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

                const hasRunningTask = data.running === true;
                sendButton.disabled = hasRunningTask;
            } catch (err) {
                console.error("Could not load messages:", err);
            }
        }

        // Polling fallback. This makes the UI work even when EventSource is flaky.
        setInterval(loadMessages, 1000);

        // Live updates. Nice when it works, but polling is the reliable fallback.
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


running = False


def broadcast(role: str, text: str) -> None:
    msg = {
        "role": role,
        "text": text,
    }

    messages.append(msg)

    dead = []
    for q in subscribers:
        try:
            q.put_nowait(msg)
        except Exception:
            dead.append(q)

    for q in dead:
        if q in subscribers:
            subscribers.remove(q)


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
                def progress(message: str) -> None:
                    broadcast("progress", message)

                final = agent.run_agentic_task(task, progress_callback=progress)
                broadcast("agent", final)
        finally:
            running = False
            broadcast("status", "idle")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    return jsonify({"started": True})


if __name__ == "__main__":
    print("Testing Ollama connection...")

    if not agent.wait_for_ollama():
        print("Warning: Ollama is not reachable yet.")

    print("Starting web app on http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, threaded=True)