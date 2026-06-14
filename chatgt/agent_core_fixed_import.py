#!/usr/bin/env python3

import json
import os
import re
import shlex
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional

import requests


ProgressCallback = Optional[Callable[[str], None]]


class OllamaAgent:
    def __init__(self, ollama_url: str | None = None, model: str | None = None):
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma4:e2b")
        self.working_dir = os.getenv("AGENT_WORKDIR", "/agent/workdir")

        self.max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "8"))
        self.max_file_read_chars = int(os.getenv("AGENT_MAX_FILE_READ_CHARS", "60000"))
        self.model_timeout_seconds = int(os.getenv("AGENT_MODEL_TIMEOUT_SECONDS", "240"))
        self.heartbeat_seconds = int(os.getenv("AGENT_HEARTBEAT_SECONDS", "5"))

        os.makedirs(self.working_dir, exist_ok=True)

    # -------------------------
    # Ollama
    # -------------------------

    def check_ollama_status(self) -> bool:
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException as exc:
            print(f"Ollama not ready: {exc}")
            return False

    def wait_for_ollama(self, max_wait_seconds: int = 60) -> bool:
        waited = 0

        while waited < max_wait_seconds:
            if self.check_ollama_status():
                return True

            print("Waiting for Ollama to start...")
            time.sleep(2)
            waited += 2

        return False

    def query_ollama(
        self,
        prompt: str,
        timeout: int | None = None,
        progress_callback: ProgressCallback = None,
    ) -> str:
        if not self.wait_for_ollama():
            return "Error: Ollama is not responding after waiting"

        timeout = timeout or self.model_timeout_seconds

        result_box = {
            "done": False,
            "response": None,
            "error": None,
        }

        def request_worker() -> None:
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=timeout,
                )

                if response.status_code != 200:
                    result_box["error"] = (
                        f"Error from Ollama {response.status_code}: {response.text}"
                    )
                    return

                result_box["response"] = response.json().get("response", "")

            except requests.exceptions.Timeout:
                result_box["error"] = "Error: Ollama request timed out"
            except requests.RequestException as exc:
                result_box["error"] = f"Error communicating with Ollama: {exc}"
            except ValueError as exc:
                result_box["error"] = f"Error parsing Ollama response: {exc}"
            finally:
                result_box["done"] = True

        thread = threading.Thread(target=request_worker, daemon=True)
        thread.start()

        elapsed = 0

        while not result_box["done"]:
            time.sleep(self.heartbeat_seconds)
            elapsed += self.heartbeat_seconds

            if result_box["done"]:
                break

            if progress_callback:
                progress_callback(f"Model is still generating... {elapsed}s")

        thread.join(timeout=1)

        if result_box["error"]:
            return str(result_box["error"])

        return str(result_box["response"] or "")

    # -------------------------
    # Paths and files
    # -------------------------

    def safe_path(self, filename: str) -> str:
        base = os.path.abspath(self.working_dir)
        path = os.path.abspath(os.path.join(base, filename))

        if path != base and not path.startswith(base + os.sep):
            raise ValueError("Refusing to access path outside working directory")

        return path

    def list_files(self, path: str = ".") -> Dict[str, object]:
        try:
            target = self.safe_path(path)

            if not os.path.exists(target):
                return {
                    "success": False,
                    "error": f"Path does not exist: {path}",
                }

            if os.path.isfile(target):
                return {
                    "success": True,
                    "files": [path],
                }

            entries = []
            for name in sorted(os.listdir(target)):
                full = os.path.join(target, name)
                rel = os.path.relpath(full, self.working_dir)

                entries.append({
                    "name": name,
                    "path": rel,
                    "type": "directory" if os.path.isdir(full) else "file",
                    "size": os.path.getsize(full) if os.path.isfile(full) else None,
                })

            return {
                "success": True,
                "path": path,
                "entries": entries,
            }

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
            }

    def read_file(self, filename: str) -> Dict[str, object]:
        try:
            path = self.safe_path(filename)

            if not os.path.isfile(path):
                return {
                    "success": False,
                    "error": f"File does not exist: {filename}",
                }

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(self.max_file_read_chars + 1)

            truncated = len(content) > self.max_file_read_chars
            content = content[: self.max_file_read_chars]

            return {
                "success": True,
                "filename": filename,
                "content": content,
                "truncated": truncated,
            }

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
            }

    def write_file(self, filename: str, content: str) -> Dict[str, object]:
        try:
            path = self.safe_path(filename)

            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            return {
                "success": True,
                "filename": filename,
                "path": path,
                "bytes": len(content.encode("utf-8")),
            }

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
            }

    def clean_file_content(self, content: str) -> str:
        content = content.strip()

        content = re.sub(
            r"^```(?:html|javascript|js|css|text|python|json)?\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )
        content = re.sub(r"\s*```$", "", content)

        return content.strip() + "\n"

    # -------------------------
    # Commands
    # -------------------------

    def is_command_allowed(self, command: str) -> tuple[bool, str]:
        command_lower = command.lower().strip()

        blocked_patterns = [
            r"\brm\b",
            r"\bmv\b.*\s/",
            r"\bchmod\b",
            r"\bchown\b",
            r"\bdd\b",
            r"\bmkfs\b",
            r"\bmount\b",
            r"\bumount\b",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bpoweroff\b",
            r"\bformat\b",
            r"\bpasswd\b",
            r"\bsu\b",
            r"\bsudo\b",
            r"\bcurl\b.*\|\s*(sh|bash)",
            r"\bwget\b.*\|\s*(sh|bash)",
            r"[&|`]",
            r"\$\(",
            r">\s*/",
        ]

        for pattern in blocked_patterns:
            if re.search(pattern, command_lower):
                return False, f"Blocked by safety rule: {pattern}"

        try:
            parts = shlex.split(command)
        except ValueError as exc:
            return False, f"Invalid command syntax: {exc}"

        if not parts:
            return False, "Empty command"

        allowed_executables = {
            "cat",
            "echo",
            "find",
            "grep",
            "head",
            "ls",
            "mkdir",
            "pwd",
            "python",
            "python3",
            "sed",
            "tail",
            "touch",
            "wc",
        }

        executable = os.path.basename(parts[0])
        if executable not in allowed_executables:
            return False, f"Command not in allowlist: {executable}"

        return True, "Allowed"

    def run_command(self, command: str) -> Dict[str, object]:
        allowed, reason = self.is_command_allowed(command)
        if not allowed:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Security violation: {reason}",
                "returncode": 1,
            }

        try:
            result = subprocess.run(
                shlex.split(command),
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout[-8000:],
                "stderr": result.stderr[-8000:],
                "returncode": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1,
            }
        except Exception as exc:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Error executing command: {exc}",
                "returncode": -1,
            }

    # -------------------------
    # Agent loop
    # -------------------------

    def extract_json_object(self, text: str) -> Dict[str, object]:
        text = text.strip()

        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])

        raise ValueError(f"No valid JSON object found in response: {text[:500]}")

    def build_agent_prompt(
        self,
        task: str,
        history: List[Dict[str, object]],
    ) -> str:
        history_text = json.dumps(history[-12:], indent=2)

        return f"""
You are a local coding agent running inside a restricted Docker work directory.

User task:
{task}

Working directory:
{self.working_dir}

Previous steps and observations:
{history_text}

You must respond with exactly one JSON object. No markdown. No explanations outside JSON.

Available actions:

1. respond
Use this for greetings, questions, explanations, or anything that does not require file/action work.
{{
  "summary": "Short progress update for the user.",
  "action": "respond",
  "message": "Direct response to the user."
}}

2. list_files
{{
  "summary": "Short progress update for the user.",
  "action": "list_files",
  "path": "."
}}

3. read_file
{{
  "summary": "Short progress update for the user.",
  "action": "read_file",
  "filename": "example.html"
}}

4. write_file
{{
  "summary": "Short progress update for the user.",
  "action": "write_file",
  "filename": "example.html",
  "content": "complete file content here"
}}

5. run_command
{{
  "summary": "Short progress update for the user.",
  "action": "run_command",
  "command": "ls -la"
}}

6. finish
Use this only after completing an agentic task.
{{
  "summary": "Short final summary for the user.",
  "action": "finish",
  "message": "Final response to the user."
}}

Rules:
- For normal conversation, use respond.
- For questions that only need an answer, use respond.
- For coding/file tasks, work step by step.
- Give a useful short summary on every iteration.
- Prefer write_file for creating or editing files.
- Do not use shell redirection, heredocs, pipes, backticks, ampersands, or destructive commands.
- If modifying an existing file, read it first unless its content is already in the history.
- For HTML tasks, write a complete standalone HTML file.
- Finish only when the task is actually complete.
- The user should always receive a useful message, not only "Task completed."
""".strip()

    def execute_action(self, action_obj: Dict[str, object]) -> Dict[str, object]:
        action = str(action_obj.get("action", "")).strip()

        if action == "respond":
            return {
                "success": True,
                "message": str(action_obj.get("message", "")),
            }

        if action == "list_files":
            return self.list_files(str(action_obj.get("path", ".")))

        if action == "read_file":
            filename = str(action_obj.get("filename", ""))
            return self.read_file(filename)

        if action == "write_file":
            filename = str(action_obj.get("filename", ""))
            content = str(action_obj.get("content", ""))
            content = self.clean_file_content(content)
            return self.write_file(filename, content)

        if action == "run_command":
            command = str(action_obj.get("command", ""))
            return self.run_command(command)

        if action == "finish":
            return {
                "success": True,
                "finished": True,
                "message": str(action_obj.get("message", action_obj.get("summary", ""))),
            }

        return {
            "success": False,
            "error": f"Unknown action: {action}",
        }

    def run_agentic_task(
        self,
        task: str,
        progress_callback: ProgressCallback = None,
    ) -> str:
        history: List[Dict[str, object]] = []

        def emit(message: str) -> None:
            if progress_callback:
                progress_callback(message)
            else:
                print(message)

        emit(f"Task started: {task}")

        for iteration in range(1, self.max_iterations + 1):
            prompt = self.build_agent_prompt(task, history)

            emit(f"{iteration}. Calling model for next action...")
            raw_response = self.query_ollama(
                prompt,
                timeout=self.model_timeout_seconds,
                progress_callback=progress_callback,
            )
            emit(f"{iteration}. Model responded.")

            if raw_response.startswith("Error"):
                emit(raw_response)
                return raw_response

            try:
                action_obj = self.extract_json_object(raw_response)
            except Exception as exc:
                observation = {
                    "success": False,
                    "error": f"Could not parse model JSON: {exc}",
                    "raw_response": raw_response[:1000],
                }

                history.append({
                    "iteration": iteration,
                    "summary": "The model returned invalid JSON. Asking it to correct itself.",
                    "action": "invalid_json",
                    "observation": observation,
                })

                emit(f"{iteration}. The model returned invalid JSON. Retrying.")
                continue

            summary = str(action_obj.get("summary", f"Iteration {iteration}"))
            action = str(action_obj.get("action", ""))

            emit(f"{iteration}. {summary}")

            if action in {"respond", "finish"}:
                observation = self.execute_action(action_obj)
                message = str(observation.get("message") or action_obj.get("message") or summary)

                history.append({
                    "iteration": iteration,
                    "summary": summary,
                    "action": action,
                    "observation": observation,
                })

                return message

            emit(f"{iteration}. Executing action: {action}")
            observation = self.execute_action(action_obj)
            emit(f"{iteration}. Action completed: {action}")

            compact_observation = dict(observation)
            if "content" in compact_observation and isinstance(compact_observation["content"], str):
                content = compact_observation["content"]
                compact_observation["content"] = content[:3000]
                compact_observation["content_truncated_for_history"] = len(content) > 3000

            history.append({
                "iteration": iteration,
                "summary": summary,
                "action": action,
                "action_input": {
                    k: v for k, v in action_obj.items()
                    if k not in {"content"}
                },
                "observation": compact_observation,
            })

        final = f"Stopped after {self.max_iterations} iterations. The task may be incomplete."
        emit(final)
        return final

    # -------------------------
    # Terminal mode
    # -------------------------

    def interactive_loop(self) -> None:
        print("Testing Ollama connection...")

        if not self.wait_for_ollama():
            print("Error: Ollama is not reachable.")
            return

        print("Ollama is ready.")
        print(f"Model: {self.model}")
        print(f"Working directory: {self.working_dir}")
        print("Type a task, or type 'exit' to quit.")

        while True:
            try:
                task = input("\nTask> ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nExiting.")
                break

            if task.lower() in {"exit", "quit", "q"}:
                break

            if not task:
                continue

            result = self.run_agentic_task(task)
            print(f"\nFinal: {result}")


if __name__ == "__main__":
    agent = OllamaAgent()

    if len(os.sys.argv) > 1:
        task_arg = " ".join(os.sys.argv[1:])
        print(agent.run_agentic_task(task_arg))
    else:
        agent.interactive_loop()