#!/usr/bin/env python3

import base64
from datetime import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional, Union
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

from agent_routing import RoutingDecision, ROLE_KEYS, route_task
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
        self.ollama_num_ctx = self.parse_optional_int(os.getenv("OLLAMA_NUM_CTX"))
        self.current_task_history_items = int(os.getenv("AGENT_CURRENT_TASK_HISTORY_ITEMS", "12"))
        self.current_task_history_content_chars = int(
            os.getenv("AGENT_CURRENT_TASK_HISTORY_CONTENT_CHARS", "24000")
        )
        self.routing_enabled = os.getenv("AGENT_ROUTING_ENABLED", "true").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.routing_quality_mode = os.getenv("AGENT_ROUTING_QUALITY", "balanced").strip().lower()
        self.routing_roles = {
            "router": os.getenv("OLLAMA_ROUTER_MODEL", "").strip(),
            "planner": os.getenv("OLLAMA_PLANNER_MODEL", "").strip(),
            "coding": os.getenv("OLLAMA_CODING_MODEL", "").strip(),
            "vision": os.getenv("OLLAMA_VISION_MODEL", "").strip(),
        }
        legacy_routing_debug_enabled = os.getenv(
            "AGENT_ROUTING_DEBUG_LOG",
            "",
        ).strip().lower() in {"1", "true", "yes", "on"}
        legacy_workflow_log_level = os.getenv("AGENT_WORKFLOW_LOG_LEVEL", "").strip().lower()
        self.run_log_level = os.getenv("AGENT_RUN_LOG_LEVEL", "").strip().lower()
        if not self.run_log_level:
            self.run_log_level = legacy_workflow_log_level or (
                "full" if legacy_routing_debug_enabled else "off"
            )
        if self.run_log_level not in {"off", "minimal", "full"}:
            self.run_log_level = "off"
        self.run_log_file = os.getenv("AGENT_RUN_LOG_FILE", ".agent_runs.jsonl").strip()
        if not self.run_log_file:
            self.run_log_file = ".agent_runs.jsonl"
        self.run_log_detail_dir = os.getenv(
            "AGENT_RUN_LOG_DETAIL_DIR",
            ".agent_run_details",
        ).strip()
        if not self.run_log_detail_dir:
            self.run_log_detail_dir = ".agent_run_details"
        self.run_log_preview_chars = int(os.getenv("AGENT_RUN_LOG_PREVIEW_CHARS", "1200"))
        self._run_log_detail_counter = 0

        self.memory_summary_file = os.path.join(self.working_dir, ".agent_memory_summary.txt")
        self.max_memory_summary_chars = int(os.getenv("AGENT_MAX_MEMORY_SUMMARY_CHARS", "12000"))
        self._playwright = None
        self._browser = None
        self._page = None
        self._output_images: List[Dict[str, str]] = []
        self.tools = build_default_tools(self)

        os.makedirs(self.working_dir, exist_ok=True)

    def parse_optional_int(self, value: object) -> int | None:
        if value is None:
            return None

        text = str(value).strip()
        if not text:
            return None

        try:
            parsed = int(text)
        except ValueError:
            return None

        return parsed if parsed > 0 else None

    def compact_content_for_history(self, content: str) -> tuple[str, bool]:
        max_chars = max(1000, self.current_task_history_content_chars)
        if len(content) <= max_chars:
            return content, False

        marker = (
            "\n\n[... middle content truncated for prompt history; "
            "the beginning and end of the content are shown ...]\n\n"
        )
        head_chars = max(1, (max_chars - len(marker)) // 2)
        tail_chars = max(1, max_chars - len(marker) - head_chars)
        return f"{content[:head_chars]}{marker}{content[-tail_chars:]}", True

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
    # Routing
    # -------------------------

    def set_routing_enabled(self, value: object) -> bool:
        if isinstance(value, bool):
            self.routing_enabled = value
        else:
            text = str(value).strip().lower()
            self.routing_enabled = text not in {"0", "false", "no", "off"}

        return self.routing_enabled

    def set_routing_quality_mode(self, value: object) -> str:
        text = str(value or "").strip().lower()

        if text not in {"economy", "balanced", "high_quality"}:
            text = "balanced"

        self.routing_quality_mode = text
        return self.routing_quality_mode

    def set_routing_roles(self, roles: Dict[str, object] | None) -> Dict[str, str]:
        roles = roles or {}

        for key in ROLE_KEYS:
            self.routing_roles[key] = str(roles.get(key, "") or "").strip()

        return self.routing_roles

    def set_run_log_level(self, value: object) -> str:
        text = str(value or "").strip().lower()
        if text not in {"off", "minimal", "full"}:
            text = "off"

        self.run_log_level = text
        return self.run_log_level

    def run_log_path(self) -> str:
        return self.safe_path(self.run_log_file)

    def run_log_detail_path(self, run_id: str | None, filename: str) -> str:
        safe_run_id = self.slugify_log_name(run_id or "global")
        return self.safe_path(os.path.join(self.run_log_detail_dir, safe_run_id, filename))

    def clear_run_logs(self) -> None:
        log_path = self.run_log_path()
        detail_path = self.safe_path(self.run_log_detail_dir)

        if os.path.isfile(log_path):
            os.remove(log_path)

        if os.path.isdir(detail_path):
            shutil.rmtree(detail_path)

    def slugify_log_name(self, value: object) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9._-]+", "-", text)
        text = text.strip("-._")
        return text[:80] or "item"

    def should_log_run_event(self, level: str = "minimal") -> bool:
        if self.run_log_level == "off":
            return False
        if level == "full" and self.run_log_level != "full":
            return False
        return True

    def append_run_log(
        self,
        scope: str,
        event: str,
        data: Dict[str, object] | None = None,
        level: str = "minimal",
        run_id: str | None = None,
    ) -> None:
        if not self.should_log_run_event(level):
            return

        try:
            path = self.run_log_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "level": level,
                "scope": scope,
                "event": event,
            }
            if run_id:
                record["run_id"] = run_id
            if data:
                record["data"] = self.compact_run_log_data(data, level, run_id, scope, event)

            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"Could not write run log: {exc}")

    def compact_run_log_data(
        self,
        value: object,
        level: str,
        run_id: str | None,
        scope: str,
        event: str,
        path: str = "data",
    ) -> object:
        if isinstance(value, dict):
            return {
                str(key): self.compact_run_log_data(
                    item,
                    level,
                    run_id,
                    scope,
                    event,
                    f"{path}.{self.slugify_log_name(key)}",
                )
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                self.compact_run_log_data(item, level, run_id, scope, event, f"{path}.{index}")
                for index, item in enumerate(value)
            ]

        if not isinstance(value, str):
            return value

        key_name = path.rsplit(".", 1)[-1]
        compact_threshold = self.run_log_preview_chars
        if key_name == "summary" or ".findings." in path:
            compact_threshold = 300

        should_externalize = (
            level == "full"
            and (
                key_name in {"prompt", "raw-output", "raw_output"}
                or len(value) > compact_threshold
            )
        )
        should_preview = should_externalize or len(value) > compact_threshold

        if not should_preview:
            return value

        preview = value[:compact_threshold]
        compacted: Dict[str, object] = {
            "preview": preview,
            "truncated": len(value) > len(preview),
            "chars": len(value),
        }

        if should_externalize:
            details_ref = self.write_run_log_detail(value, run_id, scope, event, path)
            if details_ref:
                compacted["details_ref"] = details_ref

        return compacted

    def write_run_log_detail(
        self,
        value: str,
        run_id: str | None,
        scope: str,
        event: str,
        path: str,
    ) -> str | None:
        try:
            self._run_log_detail_counter += 1
            filename = (
                f"{self._run_log_detail_counter:04d}-"
                f"{self.slugify_log_name(scope)}-"
                f"{self.slugify_log_name(event)}-"
                f"{self.slugify_log_name(path)}.txt"
            )
            detail_path = self.run_log_detail_path(run_id, filename)
            os.makedirs(os.path.dirname(detail_path), exist_ok=True)
            with open(detail_path, "w", encoding="utf-8") as f:
                f.write(value)

            return os.path.relpath(detail_path, self.working_dir).replace(os.sep, "/")
        except Exception as exc:
            print(f"Could not write run log detail: {exc}")
            return None

    def log_routing_decision(self, task: str, decision: RoutingDecision) -> None:
        self.append_run_log(
            "routing",
            "routing_decision",
            {
                "task_preview": task[:500],
                "decision": decision.to_dict(),
            },
            level="minimal",
        )

    def route_current_task(
        self,
        task: str,
        conversation_context: List[Dict[str, object]] | None = None,
        task_images: List[str] | None = None,
    ) -> RoutingDecision:
        def classifier(prompt: str, model: str) -> str:
            return self.query_ollama(
                prompt,
                timeout=self.model_timeout_seconds,
                json_mode=True,
                image_paths=task_images or [],
                model=model,
            )

        return route_task(
            task,
            default_model=self.model,
            roles=self.routing_roles,
            quality_mode=self.routing_quality_mode,
            enabled=self.routing_enabled,
            task_images=task_images or [],
            conversation_context=conversation_context or [],
            classifier=classifier,
        )

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

        summary_model = os.getenv("AGENT_MEMORY_SUMMARY_MODEL", "").strip() or None
        summary = self.query_ollama(
            prompt,
            timeout=self.model_timeout_seconds,
            model=summary_model,
        )

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
        model: str | None = None,
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
                    "model": model or self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "2048")),
                    },
                }

                if self.ollama_num_ctx:
                    payload["options"]["num_ctx"] = self.ollama_num_ctx

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

    def browser_url(self, url: str) -> str:
        value = url.strip()
        parsed = urlparse(value)

        if parsed.scheme in {"http", "https"}:
            if not parsed.netloc:
                raise ValueError("URL is missing a host")
            return value

        if parsed.scheme == "file":
            local_path = unquote(parsed.path)
            if os.name == "nt" and re.match(r"^/[A-Za-z]:/", local_path):
                local_path = local_path[1:]
            path = os.path.abspath(local_path)
            base = os.path.abspath(self.working_dir)
            if path != base and not path.startswith(base + os.sep):
                raise ValueError("Refusing to open file outside working directory")
        elif parsed.scheme:
            raise ValueError("Only http, https, file, or workspace-relative paths are supported")
        else:
            path = self.safe_path(value)

        if not os.path.isfile(path):
            raise ValueError(f"File does not exist: {url}")

        return Path(path).resolve().as_uri()

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
            browser_url = self.browser_url(url)

            page = self.ensure_browser_page()
            response = page.goto(browser_url, wait_until="domcontentloaded", timeout=self.browser_timeout_ms)
            try:
                page.wait_for_timeout(500)
            except Exception:
                pass

            snapshot = self.browser_snapshot()
            snapshot["status_code"] = response.status if response else None
            snapshot["opened_url"] = browser_url
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

    def is_structured_workflow_result(self, value: object) -> bool:
        if not isinstance(value, dict):
            return False

        status = str(value.get("status", "")).strip().lower()
        if status in {"needed_changes", "need_changes", "needs change", "needed changes"}:
            value["status"] = "needs_changes"
            status = "needs_changes"
        if status not in {"passed", "failed", "needs_changes", "blocked", "skipped"}:
            return False

        return "summary" in value and "findings" in value and "artifacts" in value

    def normalize_structured_workflow_result(self, value: Dict[str, object]) -> Dict[str, object]:
        status = str(value.get("status", "")).strip().lower()
        if status in {"needed_changes", "need_changes", "needs change", "needed changes"}:
            value["status"] = "needs_changes"
        return value

    def log_action_loop_event(
        self,
        event: str,
        data: Dict[str, object],
        run_id: str | None = None,
    ) -> None:
        self.append_run_log(
            "agent_action",
            event,
            data,
            level="full",
            run_id=run_id,
        )

    def build_structured_finalizer_prompt(
        self,
        task: str,
        history: List[Dict[str, object]],
    ) -> str:
        history_text = json.dumps(
            history[-self.current_task_history_items:],
            indent=2,
            ensure_ascii=False,
        )

        return f"""
You are finalizing a workflow step after its tool/action iteration budget was reached.

Workflow step prompt:
{task}

Tool/action history:
{history_text}

Return exactly one JSON object with this schema and no markdown:
{{
  "status": "passed|failed|needs_changes|blocked",
  "summary": "short workflow step result summary based on the evidence",
  "findings": ["specific issue, verification result, or remaining problem"],
  "artifacts": ["created, modified, inspected, or verified files"]
}}

Rules:
- If the task appears complete from the tool history, use status "passed".
- For visual, UI, HTML, canvas, animation, game, page, layout, or screenshot-related checks, browser title/text/load success alone is not visual verification; require screenshot or explicit visual evidence before passing.
- If changes were made but verification is missing or uncertain, use status "needs_changes".
- If a command/tool error prevents progress, use status "failed" or "blocked".
- Do not claim files were changed or verified unless the history supports it.
""".strip()

    def run_structured_finalizer(
        self,
        task: str,
        history: List[Dict[str, object]],
        iteration_count: int,
        selected_model: str | None,
        progress_callback: ProgressCallback,
        task_images: List[str],
        run_log_id: str | None,
        reason: str,
    ) -> str | None:
        finalizer_prompt = self.build_structured_finalizer_prompt(task, history)
        self.log_action_loop_event(
            "structured_finalizer_started",
            {
                "iterations": iteration_count,
                "reason": reason,
                "prompt": finalizer_prompt,
            },
            run_log_id,
        )
        finalizer_response = self.query_ollama(
            finalizer_prompt,
            timeout=self.model_timeout_seconds,
            progress_callback=progress_callback,
            json_mode=True,
            image_paths=task_images,
            model=selected_model,
        )
        self.log_action_loop_event(
            "structured_finalizer_finished",
            {
                "iterations": iteration_count,
                "reason": reason,
                "raw_response": finalizer_response,
            },
            run_log_id,
        )
        try:
            finalizer_obj = self.extract_json_object(finalizer_response)
            if self.is_structured_workflow_result(finalizer_obj):
                finalizer_obj = self.normalize_structured_workflow_result(finalizer_obj)
                if progress_callback:
                    progress_callback({
                        "kind": "status",
                        "text": f"Finalized structured workflow result after {iteration_count} iterations.",
                    })
                return json.dumps(finalizer_obj, ensure_ascii=False)
        except Exception:
            return None

        return None

    def build_plain_response_prompt(
        self,
        task: str,
        conversation_context: List[Dict[str, object]] | None = None,
        current_task_history: List[Dict[str, object]] | None = None,
        routing_decision: RoutingDecision | None = None,
        require_structured_result: bool = False,
    ) -> str:
        memory_summary = self.load_memory_summary()

        conversation_context = conversation_context or []
        current_task_history = current_task_history or []

        conversation_context_text = json.dumps(
            conversation_context,
            indent=2,
            ensure_ascii=False,
        )

        current_task_history_text = json.dumps(
            current_task_history[-self.current_task_history_items:],
            indent=2,
            ensure_ascii=False,
        )
        routing_text = json.dumps(
            routing_decision.to_dict() if routing_decision else {},
            indent=2,
            ensure_ascii=False,
        )

        response_rules = (
            "- Return only the structured JSON object requested by the current user message.\n"
            "- Do not wrap it in markdown.\n"
            "- Use the current task tool/action history when it contains relevant evidence."
            if require_structured_result
            else "- Answer directly in plain text.\n"
            "- Do not output JSON.\n"
            "- Use the current task tool/action history when it contains relevant results."
        )

        return f"""
You are responding directly to the user.

Long-term memory summary:
{memory_summary or "(empty)"}

Recent visible conversation context:
{conversation_context_text or "[]"}

Current task tool/action history:
{current_task_history_text or "[]"}

Routing decision:
{routing_text}

Current user message:
{task}

Instructions:
{response_rules}
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
        routing_decision: RoutingDecision | None = None,
        require_structured_result: bool = False,
    ) -> str:
        history_text = json.dumps(
            history[-self.current_task_history_items:],
            indent=2,
            ensure_ascii=False,
        )
        memory_summary = self.load_memory_summary()
        tool_prompt = render_tool_prompt(self.tools)
        tool_rules = render_tool_rules()

        conversation_context = conversation_context or []
        conversation_context_text = json.dumps(
            conversation_context,
            indent=2,
            ensure_ascii=False,
        )
        routing_text = json.dumps(
            routing_decision.to_dict() if routing_decision else {},
            indent=2,
            ensure_ascii=False,
        )

        structured_rules = ""
        if require_structured_result:
            structured_rules = """
- This is a workflow step. Tools are available, but tool output is evidence, not the final answer.
- Always respond with an action object. Do not return the workflow result object directly.
- Do not finish by simply returning file contents, command output, or browser text.
- To complete this workflow step, use action "finish".
- The finish.message value must be the workflow result object requested in the current workflow step prompt.
""".rstrip()

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

Images available to the model, including user uploads and screenshots produced during this task:
{json.dumps(task_images or [], indent=2, ensure_ascii=False)}

Routing decision:
{routing_text}

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
- For HTML, canvas, animation, game, or visual UI checks, browser DOM text/title alone is not enough. Use screenshots or explicit visual evidence when visual output matters.
- Do not mark a visual check passed without screenshot or explicit visual evidence.
- This local-only v1 has no bitmap image-generation tool. If the routing decision says task_type is image_generation, create code-native visual output only when the user asked for a file/page; otherwise explain that local image generation is not configured.
{structured_rules}
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
        selected_model: str | None = None,
        routing_decision: RoutingDecision | None = None,
        require_structured_result: bool = False,
        max_iterations: int | None = None,
        run_log_id: str | None = None,
    ) -> str:
        history: List[Dict[str, object]] = []
        successful_action_counts: Dict[str, int] = {}
        task_images = list(task_images or [])
        evidence_images: List[str] = []
        iteration_limit = max_iterations if max_iterations and max_iterations > 0 else self.max_iterations
        self._output_images = []

        def model_images() -> List[str]:
            images: List[str] = []
            for image in [*task_images, *evidence_images]:
                if image and image not in images:
                    images.append(image)
            return images

        def emit(event: ProgressEvent) -> None:
            if progress_callback:
                progress_callback(event)
            else:
                print(event if isinstance(event, str) else json.dumps(event, indent=2))

        emit({
            "kind": "status",
            "text": f"Task started: {task}",
        })

        self.log_action_loop_event(
            "task_started",
            {
                "task_preview": task[:500],
                "selected_model": selected_model,
                "require_structured_result": require_structured_result,
                "max_iterations": iteration_limit,
            },
            run_log_id,
        )

        for iteration in range(1, iteration_limit + 1):
            prompt = self.build_agent_prompt(
                task,
                history,
                conversation_context=conversation_context,
                task_images=model_images(),
                routing_decision=routing_decision,
                require_structured_result=require_structured_result,
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
                image_paths=model_images(),
                model=selected_model,
            )

            emit({
                "kind": "model_output",
                "iteration": iteration,
                "text": raw_response,
            })
            self.log_action_loop_event(
                "model_output",
                {
                    "iteration": iteration,
                    "selected_model": selected_model,
                    "image_paths": model_images(),
                    "raw_response": raw_response,
                },
                run_log_id,
            )

            if raw_response.startswith("Error"):
                emit({
                    "kind": "error",
                    "iteration": iteration,
                    "text": raw_response,
                })
                self.log_action_loop_event(
                    "model_error",
                    {
                        "iteration": iteration,
                        "error": raw_response,
                    },
                    run_log_id,
                )
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
                self.log_action_loop_event(
                    "invalid_json",
                    {
                        "iteration": iteration,
                        "raw_response": raw_response,
                        "observation": observation,
                    },
                    run_log_id,
                )
                continue

            if require_structured_result and self.is_structured_workflow_result(action_obj):
                action_obj = self.normalize_structured_workflow_result(action_obj)
                emit({
                    "kind": "status",
                    "iteration": iteration,
                    "text": "Accepted direct structured workflow result.",
                })
                self.log_action_loop_event(
                    "direct_structured_result",
                    {
                        "iteration": iteration,
                        "result": action_obj,
                    },
                    run_log_id,
                )
                return json.dumps(action_obj, ensure_ascii=False)

            summary = str(action_obj.get("summary", f"Iteration {iteration}"))
            action = str(action_obj.get("action", ""))
            tool = self.tools.get(action)

            emit({
                "kind": "summary",
                "iteration": iteration,
                "action": action,
                "text": summary,
            })
            self.log_action_loop_event(
                "action_selected",
                {
                    "iteration": iteration,
                    "summary": summary,
                    "action": action,
                    "action_input": {
                        k: v for k, v in action_obj.items()
                        if k not in {"content"}
                    },
                },
                run_log_id,
            )

            observation = self.execute_action(action_obj)
            if action == "browser_screenshot" and observation.get("success"):
                screenshot_filename = str(observation.get("filename", "")).strip()
                if screenshot_filename and screenshot_filename not in evidence_images:
                    evidence_images.append(screenshot_filename)

            emit({
                "kind": "observation",
                "iteration": iteration,
                "action": action,
                "text": json.dumps(observation, indent=2),
            })
            self.log_action_loop_event(
                "tool_observation",
                {
                    "iteration": iteration,
                    "action": action,
                    "observation": observation,
                },
                run_log_id,
            )

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
                    routing_decision=routing_decision,
                    require_structured_result=require_structured_result,
                )

                message = self.query_ollama(
                    plain_prompt,
                    timeout=self.model_timeout_seconds,
                    progress_callback=progress_callback,
                    json_mode=False,
                    image_paths=model_images(),
                    model=selected_model,
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

                self.log_action_loop_event(
                    "respond_finished",
                    {
                        "iteration": iteration,
                        "message": message,
                    },
                    run_log_id,
                )
                return message

            if action == "finish":
                message_value = observation.get("message")
                if message_value is None or message_value == "":
                    message_value = action_obj.get("message", summary)
                message = (
                    json.dumps(message_value, ensure_ascii=False)
                    if isinstance(message_value, (dict, list))
                    else str(message_value)
                )

                history.append({
                    "iteration": iteration,
                    "summary": summary,
                    "action": action,
                    "observation": observation,
                })

                self.log_action_loop_event(
                    "finish_action",
                    {
                        "iteration": iteration,
                        "message": message,
                    },
                    run_log_id,
                )
                return message

            compact_observation = dict(observation)
            if "content" in compact_observation and isinstance(compact_observation["content"], str):
                content = compact_observation["content"]
                compacted_content, content_truncated = self.compact_content_for_history(content)
                compact_observation["content"] = compacted_content
                compact_observation["content_truncated_for_history"] = content_truncated

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
                    require_structured_result
                    and successful_action_counts[action_signature] >= 2
                ):
                    self.log_action_loop_event(
                        "repeat_action_finalizer_triggered",
                        {
                            "iteration": iteration,
                            "action": action,
                            "repeat_count": successful_action_counts[action_signature],
                        },
                        run_log_id,
                    )
                    finalized = self.run_structured_finalizer(
                        task,
                        history,
                        iteration,
                        selected_model,
                        progress_callback,
                        model_images(),
                        run_log_id,
                        reason=f"repeated successful action: {action}",
                    )
                    if finalized is not None:
                        return finalized

                if (
                    tool
                    and tool.direct_return_on_repeat
                    and successful_action_counts[action_signature] >= 2
                    and not require_structured_result
                ):
                    message = self.format_observation_for_user(action, observation)
                    self.log_action_loop_event(
                        "direct_return_on_repeat",
                        {
                            "iteration": iteration,
                            "action": action,
                            "message": message,
                        },
                        run_log_id,
                    )
                    return message

            if (
                tool
                and tool.direct_return_phrases
                and observation.get("success")
                and not require_structured_result
            ):
                task_lower = task.lower()

                if any(phrase in task_lower for phrase in tool.direct_return_phrases):
                    message = self.format_observation_for_user(action, observation)
                    self.log_action_loop_event(
                        "direct_return_phrase",
                        {
                            "iteration": iteration,
                            "action": action,
                            "message": message,
                        },
                        run_log_id,
                    )
                    return message

            if tool and tool.continue_after_success and observation.get("success"):
                emit({
                    "kind": "status",
                    "iteration": iteration,
                    "text": tool.verification_hint or "Continuing after tool result...",
                })
                continue

        if require_structured_result:
            finalized = self.run_structured_finalizer(
                task,
                history,
                iteration_limit,
                selected_model,
                progress_callback,
                model_images(),
                run_log_id,
                reason="iteration limit reached",
            )
            if finalized is not None:
                return finalized

        final = f"Stopped after {iteration_limit} iterations. The task may be incomplete."
        emit({
            "kind": "status",
            "text": final,
        })
        self.log_action_loop_event(
            "iteration_limit_reached",
            {
                "iterations": iteration_limit,
                "message": final,
            },
            run_log_id,
        )
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
