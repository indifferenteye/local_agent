#!/usr/bin/env python3

import base64
import json
import os
import re
import shlex
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional, Union
from urllib.parse import urlparse

import requests

from agent_tools import (
    build_default_tools,
    default_formatter,
    render_tool_prompt,
    render_tool_rules,
)


ProgressEvent = Union[str, Dict[str, object]]
ProgressCallback = Optional[Callable[[ProgressEvent], None]]


class OllamaAgent:
    def __init__(self, ollama_url: str | None = None, model: str | None = None):
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma4:e2b")
        self.think_mode = self.parse_think_mode(os.getenv("OLLAMA_THINK", ""))
        self.working_dir = os.getenv("AGENT_WORKDIR", "/agent/workdir")

        self.max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "8"))
        self.max_file_read_chars = int(os.getenv("AGENT_MAX_FILE_READ_CHARS", "60000"))
        self.max_url_read_chars = int(os.getenv("AGENT_MAX_URL_READ_CHARS", "60000"))
        self.max_browser_text_chars = int(os.getenv("AGENT_MAX_BROWSER_TEXT_CHARS", "12000"))
        self.browser_timeout_ms = int(os.getenv("AGENT_BROWSER_TIMEOUT_MS", "20000"))
        self.model_timeout_seconds = int(os.getenv("AGENT_MODEL_TIMEOUT_SECONDS", "240"))
        self.heartbeat_seconds = int(os.getenv("AGENT_HEARTBEAT_SECONDS", "5"))

        self.memory_summary_file = os.path.join(self.working_dir, ".agent_memory_summary.txt")
        self.max_memory_summary_chars = int(os.getenv("AGENT_MAX_MEMORY_SUMMARY_CHARS", "12000"))
        self._playwright = None
        self._browser = None
        self._page = None
        self._output_images: List[Dict[str, str]] = []
        self.tools = build_default_tools(self)

        os.makedirs(self.working_dir, exist_ok=True)

    # -------------------------
    # Thinking mode
    # -------------------------

    def parse_think_mode(self, value: object) -> object:
        if value is None:
            return None

        if isinstance(value, bool):
            return value

        text = str(value).strip().lower()

        if text in {"", "default", "none", "null"}:
            return None

        if text in {"true", "on", "yes", "think"}:
            return True

        if text in {"false", "off", "no", "nothink"}:
            return False

        if text in {"low", "medium", "high"}:
            return text

        return None

    def set_think_mode(self, value: object) -> object:
        self.think_mode = self.parse_think_mode(value)
        return self.think_mode

    def get_think_mode_label(self) -> str:
        if self.think_mode is None:
            return "default"

        if self.think_mode is True:
            return "think"

        if self.think_mode is False:
            return "nothink"

        return str(self.think_mode)

    # -------------------------
    # Memory
    # -------------------------

    def load_memory_summary(self) -> str:
        try:
            if os.path.exists(self.memory_summary_file):
                with open(self.memory_summary_file, "r", encoding="utf-8") as f:
                    return f.read().strip()
        except Exception:
            pass

        return ""

    def save_memory_summary(self, summary: str) -> None:
        summary = summary.strip()

        if len(summary) > self.max_memory_summary_chars:
            summary = summary[-self.max_memory_summary_chars:]

        os.makedirs(os.path.dirname(self.memory_summary_file), exist_ok=True)

        with open(self.memory_summary_file, "w", encoding="utf-8") as f:
            f.write(summary + "\n")

    def summarize_conversation_memory(
        self,
        existing_summary: str,
        messages_to_summarize: List[Dict[str, object]],
    ) -> str:
        compact_messages = []

        for msg in messages_to_summarize:
            role = str(msg.get("role", ""))
            text = str(msg.get("text", ""))

            if role in {"progress", "status"}:
                continue

            if not text.strip():
                continue

            compact_messages.append({
                "role": role,
                "text": text[:3000],
            })

        if not compact_messages:
            return existing_summary

        prompt = f"""
You are maintaining long-term memory for a local coding agent.

Existing memory summary:
{existing_summary or "(empty)"}

New conversation messages to integrate:
{json.dumps(compact_messages, indent=2, ensure_ascii=False)}

Create an updated compact memory summary.

Rules:
- Keep only useful durable information.
- Remember user preferences, ongoing project details, important files created or edited, important bugs/fixes, and decisions.
- Do not include raw model JSON.
- Do not include repetitive progress logs.
- Do not include huge file contents.
- Keep it concise but useful for future tasks.
- Return only the updated memory summary as plain text.
""".strip()

        summary = self.query_ollama(prompt, timeout=self.model_timeout_seconds)

        if summary.startswith("Error"):
            return existing_summary

        summary = summary.strip()

        if len(summary) > self.max_memory_summary_chars:
            summary = summary[-self.max_memory_summary_chars:]

        return summary

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

    def list_installed_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)

            if response.status_code != 200:
                return []

            data = response.json()
            models = data.get("models", [])

            names = []
            for model in models:
                name = model.get("name") or model.get("model")
                if name:
                    names.append(str(name))

            return sorted(set(names))

        except Exception:
            return []

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
        json_mode: bool = False,
        image_paths: List[str] | None = None,
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
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "2048")),
                    },
                }

                if json_mode:
                    payload["format"] = "json"

                if self.think_mode is not None:
                    payload["think"] = self.think_mode

                encoded_images = self.encode_images(image_paths or [])
                if encoded_images:
                    payload["images"] = encoded_images

                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
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
                progress_callback({
                    "kind": "heartbeat",
                    "text": f"Model is still generating... {elapsed}s",
                })

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

    def encode_images(self, filenames: List[str]) -> List[str]:
        encoded = []

        for filename in filenames:
            try:
                path = self.safe_path(filename)

                if not os.path.isfile(path):
                    continue

                if os.path.splitext(path.lower())[1] not in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                    ".bmp",
                }:
                    continue

                with open(path, "rb") as f:
                    encoded.append(base64.b64encode(f.read()).decode("ascii"))

            except Exception:
                continue

        return encoded

    def image_attachment(self, filename: str, label: str | None = None) -> Dict[str, str]:
        path = self.safe_path(filename)

        if not os.path.isfile(path):
            raise FileNotFoundError(filename)

        if os.path.splitext(path.lower())[1] not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".webp",
            ".bmp",
        }:
            raise ValueError("Unsupported image type")

        rel = os.path.relpath(path, self.working_dir).replace("\\", "/")

        return {
            "filename": rel,
            "label": label or os.path.basename(rel),
            "url": f"/workdir-image/{rel}",
        }

    def send_image(self, filename: str, label: str | None = None) -> Dict[str, object]:
        try:
            image = self.image_attachment(filename, label)
            self._output_images.append(image)

            return {
                "success": True,
                "image": image,
            }

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "filename": filename,
            }

    def consume_output_images(self) -> List[Dict[str, str]]:
        images = list(self._output_images)
        self._output_images.clear()
        return images

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

                if name in {
                    ".agent_sessions.json",
                    ".agent_memory_summary.txt",
                    ".agent_memory_state.json",
                    ".agent_settings.json",
                }:
                    continue

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
            content = content[:self.max_file_read_chars]

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
    # Web fetches
    # -------------------------

    def fetch_url(self, url: str) -> Dict[str, object]:
        try:
            url = url.strip()
            parsed = urlparse(url)

            if parsed.scheme not in {"http", "https"}:
                return {
                    "success": False,
                    "error": "Only http and https URLs are supported",
                    "url": url,
                }

            if not parsed.netloc:
                return {
                    "success": False,
                    "error": "URL is missing a host",
                    "url": url,
                }

            response = requests.get(
                url,
                headers={
                    "User-Agent": "local-ollama-agent/1.0",
                },
                timeout=20,
                allow_redirects=True,
            )

            content = response.text
            truncated = len(content) > self.max_url_read_chars
            content = content[:self.max_url_read_chars]

            return {
                "success": True,
                "url": url,
                "final_url": response.url,
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "content": content,
                "truncated": truncated,
            }

        except requests.RequestException as exc:
            return {
                "success": False,
                "error": f"Error fetching URL: {exc}",
                "url": url,
            }

    def simple_curl_url(self, command: str) -> str | None:
        try:
            parts = shlex.split(command)
        except ValueError:
            return None

        if not parts or os.path.basename(parts[0]) != "curl":
            return None

        urls = []
        allowed_flags = {"-s", "-S", "-L", "-I", "-i", "-k", "--silent", "--show-error", "--location"}

        for part in parts[1:]:
            if part.startswith("-"):
                if part in allowed_flags:
                    continue

                if part.startswith("-") and set(part.lstrip("-")) <= {"s", "S", "L", "I", "i", "k"}:
                    continue

                return None

            urls.append(part)

        if len(urls) != 1:
            return None

        return urls[0]

    # -------------------------
    # Browser automation
    # -------------------------

    def ensure_browser_page(self):
        try:
            if self._page is not None:
                return self._page

            from playwright.sync_api import sync_playwright

            if self._playwright is None:
                self._playwright = sync_playwright().start()

            if self._browser is None:
                self._browser = self._playwright.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage"],
                )

            self._page = self._browser.new_page()
            self._page.set_default_timeout(self.browser_timeout_ms)
            return self._page

        except Exception as exc:
            raise RuntimeError(
                "Browser support is unavailable. Rebuild the Docker image after "
                "installing Playwright/Chromium."
            ) from exc

    def browser_snapshot(self) -> Dict[str, object]:
        try:
            page = self.ensure_browser_page()

            title = page.title()
            url = page.url

            try:
                text = page.locator("body").inner_text(timeout=3000)
            except Exception:
                text = ""

            truncated = len(text) > self.max_browser_text_chars
            text = text[:self.max_browser_text_chars]

            links = page.locator("a").evaluate_all(
                """
                els => els.slice(0, 30).map(el => ({
                    text: (el.innerText || el.textContent || '').trim().slice(0, 120),
                    href: el.href || ''
                })).filter(item => item.text || item.href)
                """
            )
            buttons = page.locator("button, input[type=button], input[type=submit]").evaluate_all(
                """
                els => els.slice(0, 30).map(el => ({
                    text: (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().slice(0, 120),
                    type: el.tagName.toLowerCase()
                })).filter(item => item.text)
                """
            )

            return {
                "success": True,
                "url": url,
                "title": title,
                "text": text,
                "text_truncated": truncated,
                "links": links,
                "buttons": buttons,
            }

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
            }

    def browser_open(self, url: str) -> Dict[str, object]:
        try:
            url = url.strip()
            parsed = urlparse(url)

            if parsed.scheme not in {"http", "https"}:
                return {
                    "success": False,
                    "error": "Only http and https URLs are supported",
                    "url": url,
                }

            if not parsed.netloc:
                return {
                    "success": False,
                    "error": "URL is missing a host",
                    "url": url,
                }

            page = self.ensure_browser_page()
            response = page.goto(url, wait_until="domcontentloaded", timeout=self.browser_timeout_ms)

            snapshot = self.browser_snapshot()
            snapshot["status_code"] = response.status if response else None
            return snapshot

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "url": url,
            }

    def browser_click(self, selector: str) -> Dict[str, object]:
        try:
            page = self.ensure_browser_page()
            page.click(selector, timeout=self.browser_timeout_ms)

            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass

            return self.browser_snapshot()

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "selector": selector,
            }

    def browser_type(self, selector: str, text: str) -> Dict[str, object]:
        try:
            page = self.ensure_browser_page()
            page.fill(selector, text, timeout=self.browser_timeout_ms)
            return self.browser_snapshot()

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "selector": selector,
            }

    def browser_screenshot(self, filename: str = "browser-screenshot.png") -> Dict[str, object]:
        try:
            page = self.ensure_browser_page()
            path = self.safe_path(filename)

            os.makedirs(os.path.dirname(path), exist_ok=True)
            page.screenshot(path=path, full_page=True)

            return {
                "success": True,
                "filename": filename,
                "path": path,
                "url": page.url,
                "title": page.title(),
            }

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "filename": filename,
            }

    def browser_close(self) -> Dict[str, object]:
        try:
            if self._page is not None:
                self._page.close()
                self._page = None

            if self._browser is not None:
                self._browser.close()
                self._browser = None

            if self._playwright is not None:
                self._playwright.stop()
                self._playwright = None

            return {
                "success": True,
                "message": "Browser closed",
            }

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
            }

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
        curl_url = self.simple_curl_url(command)
        if curl_url:
            fetched = self.fetch_url(curl_url)
            if not fetched.get("success"):
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": str(fetched.get("error", "URL fetch failed")),
                    "returncode": 1,
                }

            content = str(fetched.get("content", ""))
            return {
                "success": True,
                "stdout": content[-8000:],
                "stderr": "",
                "returncode": 0,
                "url": fetched.get("url"),
                "final_url": fetched.get("final_url"),
                "status_code": fetched.get("status_code"),
                "content_type": fetched.get("content_type"),
                "truncated": fetched.get("truncated"),
            }

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
            return json.loads(text[start:end + 1])

        raise ValueError(f"No valid JSON object found in response: {text[:500]}")

    def format_observation_for_user(self, action: str, observation: Dict[str, object]) -> str:
        if not observation.get("success"):
            return f"Action failed: {observation.get('error', 'Unknown error')}"

        tool = self.tools.get(action)
        formatter = tool.formatter if tool and tool.formatter else default_formatter
        return formatter(observation)

    def build_plain_response_prompt(
        self,
        task: str,
        conversation_context: List[Dict[str, object]] | None = None,
        current_task_history: List[Dict[str, object]] | None = None,
    ) -> str:
        memory_summary = self.load_memory_summary()

        conversation_context = conversation_context or []
        current_task_history = current_task_history or []

        conversation_context_text = json.dumps(
            conversation_context[-20:],
            indent=2,
            ensure_ascii=False,
        )

        current_task_history_text = json.dumps(
            current_task_history[-12:],
            indent=2,
            ensure_ascii=False,
        )

        return f"""
You are responding directly to the user.

Long-term memory summary:
{memory_summary or "(empty)"}

Recent visible conversation context:
{conversation_context_text or "[]"}

Current task tool/action history:
{current_task_history_text or "[]"}

Current user message:
{task}

Instructions:
- Answer directly in plain text.
- Do not output JSON.
- Use the current task tool/action history when it contains relevant results.
- If a tool result says files were found, report those files. Do not claim you cannot access files.
- Use recent conversation context to resolve references like "it", "that", "continue", "make it rhyme", "rewrite it", etc.
- If the user asks to transform the previous answer, transform the previous agent answer.
- Keep the answer complete.
""".strip()

    def build_agent_prompt(
        self,
        task: str,
        history: List[Dict[str, object]],
        conversation_context: List[Dict[str, object]] | None = None,
        task_images: List[str] | None = None,
    ) -> str:
        history_text = json.dumps(history[-12:], indent=2, ensure_ascii=False)
        memory_summary = self.load_memory_summary()
        tool_prompt = render_tool_prompt(self.tools)
        tool_rules = render_tool_rules()

        conversation_context = conversation_context or []
        conversation_context_text = json.dumps(
            conversation_context[-20:],
            indent=2,
            ensure_ascii=False,
        )

        return f"""
You are a local coding agent running inside a restricted Docker work directory.

Long-term memory summary:
{memory_summary or "(empty)"}

Recent visible conversation context:
{conversation_context_text or "[]"}

Current user task:
{task}

Working directory:
{self.working_dir}

Previous steps and observations for this current task:
{history_text}

Uploaded images available to the model:
{json.dumps(task_images or [], indent=2, ensure_ascii=False)}

You must respond with exactly one JSON object. No markdown. No explanations outside JSON.

Available actions:

{tool_prompt}

Rules:
- Use the recent visible conversation context to resolve references like "it", "that", "make it better", "continue", "change the last answer", etc.
- If the user asks to transform or continue the previous answer, use the previous agent response from conversation context.
- Use the long-term memory only when relevant.
{tool_rules}
- Give a useful short summary on every iteration.
- If modifying an existing file, read it first unless its content is already in the task history.
- For HTML tasks, write a complete standalone HTML file.
- Finish only when the task is actually complete.
- The user should always receive a useful message, not only "Task completed."
""".strip()

    def execute_action(self, action_obj: Dict[str, object]) -> Dict[str, object]:
        action = str(action_obj.get("action", "")).strip()
        tool = self.tools.get(action)

        if tool:
            return tool.handler(action_obj)

        return {
            "success": False,
            "error": f"Unknown action: {action}",
        }

    def run_agentic_task(
        self,
        task: str,
        progress_callback: ProgressCallback = None,
        conversation_context: List[Dict[str, object]] | None = None,
        task_images: List[str] | None = None,
    ) -> str:
        history: List[Dict[str, object]] = []
        successful_action_counts: Dict[str, int] = {}
        task_images = task_images or []
        self._output_images = []

        def emit(event: ProgressEvent) -> None:
            if progress_callback:
                progress_callback(event)
            else:
                print(event if isinstance(event, str) else json.dumps(event, indent=2))

        emit({
            "kind": "status",
            "text": f"Task started: {task}",
        })

        for iteration in range(1, self.max_iterations + 1):
            prompt = self.build_agent_prompt(
                task,
                history,
                conversation_context=conversation_context,
                task_images=task_images,
            )

            emit({
                "kind": "status",
                "iteration": iteration,
                "text": f"{iteration}. Calling model for next action...",
            })

            raw_response = self.query_ollama(
                prompt,
                timeout=self.model_timeout_seconds,
                progress_callback=progress_callback,
                json_mode=True,
                image_paths=task_images,
            )

            emit({
                "kind": "model_output",
                "iteration": iteration,
                "text": raw_response,
            })

            if raw_response.startswith("Error"):
                emit({
                    "kind": "error",
                    "iteration": iteration,
                    "text": raw_response,
                })
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

                emit({
                    "kind": "observation",
                    "iteration": iteration,
                    "summary": "Invalid JSON",
                    "text": json.dumps(observation, indent=2),
                })
                continue

            summary = str(action_obj.get("summary", f"Iteration {iteration}"))
            action = str(action_obj.get("action", ""))
            tool = self.tools.get(action)

            emit({
                "kind": "summary",
                "iteration": iteration,
                "action": action,
                "text": summary,
            })

            observation = self.execute_action(action_obj)

            emit({
                "kind": "observation",
                "iteration": iteration,
                "action": action,
                "text": json.dumps(observation, indent=2),
            })

            if action == "respond":
                emit({
                    "kind": "status",
                    "iteration": iteration,
                    "text": "Generating direct response...",
                })

                plain_prompt = self.build_plain_response_prompt(
                    task,
                    conversation_context=conversation_context,
                    current_task_history=history,
                )

                message = self.query_ollama(
                    plain_prompt,
                    timeout=self.model_timeout_seconds,
                    progress_callback=progress_callback,
                    json_mode=False,
                    image_paths=task_images,
                )

                history.append({
                    "iteration": iteration,
                    "summary": summary,
                    "action": action,
                    "observation": {
                        "success": True,
                        "message": message[:1000],
                    },
                })

                return message

            if action == "finish":
                message = str(observation.get("message") or action_obj.get("message") or summary)

                history.append({
                    "iteration": iteration,
                    "summary": summary,
                    "action": action,
                    "observation": observation,
                })

                return message

            compact_observation = dict(observation)
            if "content" in compact_observation and isinstance(compact_observation["content"], str):
                content = compact_observation["content"]
                compact_observation["content"] = content[:3000]
                compact_observation["content_truncated_for_history"] = len(content) > 3000

            if not observation.get("success"):
                compact_observation["policy_hint"] = (
                    "The action failed. Use the error details to choose a corrected next action. "
                    "Do not repeat the same failing action without changing the input."
                )
            elif tool and tool.verification_hint:
                compact_observation["policy_hint"] = tool.verification_hint

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
            action_signature = json.dumps(
                {
                    "action": action,
                    "input": {
                        k: v for k, v in action_obj.items()
                        if k not in {"summary", "content"}
                    },
                },
                sort_keys=True,
                ensure_ascii=False,
            )

            if observation.get("success"):
                successful_action_counts[action_signature] = (
                    successful_action_counts.get(action_signature, 0) + 1
                )

                if (
                    tool
                    and tool.direct_return_on_repeat
                    and successful_action_counts[action_signature] >= 2
                ):
                    return self.format_observation_for_user(action, observation)

            if tool and tool.direct_return_phrases and observation.get("success"):
                task_lower = task.lower()

                if any(phrase in task_lower for phrase in tool.direct_return_phrases):
                    return self.format_observation_for_user(action, observation)

            if tool and tool.continue_after_success and observation.get("success"):
                emit({
                    "kind": "status",
                    "iteration": iteration,
                    "text": tool.verification_hint or "Continuing after tool result...",
                })
                continue

        final = f"Stopped after {self.max_iterations} iterations. The task may be incomplete."
        emit({
            "kind": "status",
            "text": final,
        })
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
        print(f"Thinking mode: {self.get_think_mode_label()}")
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
