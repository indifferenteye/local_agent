#!/usr/bin/env python3

import logging

from flask import Flask

import app_state as state
from persistence import load_messages, load_settings
from routes import register_routes


app = Flask(__name__)
register_routes(app)

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

load_messages()
load_settings()


if __name__ == "__main__":
    print("Testing Ollama connection...")

    if not state.agent.wait_for_ollama():
        print("Warning: Ollama is not reachable yet.")

    print(f"Session file: {state.SESSION_FILE}")
    print(f"Settings file: {state.SETTINGS_FILE}")
    print(f"Memory summary file: {state.agent.memory_summary_file}")
    print(f"Memory state file: {state.MEMORY_STATE_FILE}")
    print("Starting web app on http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, threaded=True)
