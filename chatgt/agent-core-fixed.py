#!/usr/bin/env python3

import json
import os
import re
import shlex
import subprocess
import time
from typing import Dict, List

import requests


class OllamaAgent:
    def __init__(self, ollama_url: str | None = None, model: str | None = None):
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma4:e2b")

        # Important: use /agent so created files appear in your mounted Windows folder
        self.working_dir = os.getenv("AGENT_WORKDIR", "/agent")
        os.makedirs(self.working_dir, exist_ok=True)

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

    def query_ollama(self, prompt: str) -> str:
        if not self.wait_for_ollama():
            return "Error: Ollama is not responding after waiting"

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=180,
            )

            if response.status_code != 200:
                return f"Error from Ollama {response.status_code}: {response.text}"

            return response.json().get("response", "")

        except requests.exceptions.Timeout:
            return "Error: Ollama request timed out"
        except requests.RequestException as exc:
            return f"Error communicating with Ollama: {exc}"
        except ValueError as exc:
            return f"Error parsing Ollama response: {exc}"

    def safe_path(self, filename: str) -> str:
        base = os.path.abspath(self.working_dir)
        path = os.path.abspath(os.path.join(base, filename))

        if path != base and not path.startswith(base + os.sep):
            raise ValueError("Refusing to access path outside working directory")

        return path

    def write_file(self, filename: str, content: str) -> Dict[str, object]:
        try:
            path = self.safe_path(filename)

            parent = os.path.dirname(path)
            os.makedirs(parent, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            return {
                "success": True,
                "path": path,
                "bytes": len(content.encode("utf-8")),
            }

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
            }

    def extract_filename_from_task(self, task: str, default: str = "output.html") -> str:
        task_lower = task.lower()

        match = re.search(r"(?:named|called|file named|file called)\s+([a-zA-Z0-9_.-]+\.[a-zA-Z0-9]+)", task)
        if match:
            return match.group(1)

        match = re.search(r"([a-zA-Z0-9_.-]+\.html)", task)
        if match:
            return match.group(1)

        if "donut" in task_lower:
            return "donut.html"

        return default

    def create_html_file(self, task: str) -> str:
        filename = self.extract_filename_from_task(task, "output.html")

        prompt = f"""
Create the complete contents of a single standalone HTML file for this request:

{task}

Requirements:
- Return only the HTML content.
- Do not wrap it in markdown.
- Do not include explanations.
- Include CSS and JavaScript inside the HTML file.
- The file must work when opened directly in a browser.
""".strip()

        html = self.query_ollama(prompt)

        if html.startswith("Error"):
            return html

        html = self.clean_model_file_content(html)
        result = self.write_file(filename, html)

        return (
            f"HTML file creation result:\n"
            f"{json.dumps(result, indent=2)}"
        )

    def clean_model_file_content(self, content: str) -> str:
        content = content.strip()

        # Remove markdown fences if the model still adds them
        content = re.sub(r"^```(?:html|javascript|js|css|text)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)

        return content.strip() + "\n"

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

    def execute_command(self, command: str) -> Dict[str, object]:
        allowed, reason = self.is_command_allowed(command)
        if not allowed:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Security violation: {reason}",
                "returncode": 1,
            }

        try:
            print(f"Executing command: {command}")

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
                "stdout": result.stdout,
                "stderr": result.stderr,
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

    def extract_commands(self, response_text: str) -> List[str]:
        commands: List[str] = []

        code_blocks = re.findall(
            r"```(?:bash|shell|sh)?\s*(.*?)```",
            response_text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        for block in code_blocks:
            for line in block.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                commands.append(line[2:].strip() if line.startswith("$ ") else line)

        if commands:
            return commands

        for line in response_text.splitlines():
            line = line.strip()
            if line.startswith("$ "):
                commands.append(line[2:].strip())

        return commands

    def extract_and_execute_commands(self, response_text: str) -> List[Dict[str, object]]:
        results = []

        for command in self.extract_commands(response_text):
            results.append({
                "command": command,
                "result": self.execute_command(command),
            })

        return results

    def run_task(self, task: str) -> str:
        task_lower = task.lower()

        # File-generation shortcut
        if "html" in task_lower and "file" in task_lower:
            return self.create_html_file(task)

        prompt = f"""
You are an assistant that can produce safe shell commands for a restricted Linux work directory.

The user wants you to:
{task}

Rules:
- Only output commands when they are necessary.
- Use only safe single-line commands.
- Put commands in a fenced shell block.
- Do not use destructive commands.
- Do not use shell redirection, heredocs, pipes, backticks, ampersands, or command substitution.
- For creating or editing files, prefer Python scripts or explain what should be done.
""".strip()

        print(f"Processing task: {task}")

        response = self.query_ollama(prompt)

        print("Response from Ollama:")
        print(response)

        if response.startswith("Error"):
            return response

        results = self.extract_and_execute_commands(response)

        return (
            f"Task completed. Response: {response}\n\n"
            f"Command execution results:\n{json.dumps(results, indent=2)}"
        )

    def process_command(self, command: str) -> str:
        result = self.execute_command(command)
        return f"Command execution result:\n{json.dumps(result, indent=2)}"

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

            print(self.run_task(task))


if __name__ == "__main__":
    agent = OllamaAgent()

    if len(os.sys.argv) > 1:
        print(agent.run_task(" ".join(os.sys.argv[1:])))
    else:
        agent.interactive_loop()