#!/usr/bin/env python3

import os
import json
import sys
import tempfile
import types
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHATGT = os.path.join(ROOT, "chatgt")
if CHATGT not in sys.path:
    sys.path.insert(0, CHATGT)

sys.modules.setdefault("requests", types.SimpleNamespace(get=None))

from agent_core import OllamaAgent


class HistoryCompactionTests(unittest.TestCase):
    def make_agent(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        old_workdir = os.environ.get("AGENT_WORKDIR")
        os.environ["AGENT_WORKDIR"] = tmpdir.name
        try:
            return OllamaAgent()
        finally:
            if old_workdir is None:
                os.environ.pop("AGENT_WORKDIR", None)
            else:
                os.environ["AGENT_WORKDIR"] = old_workdir

    def test_medium_content_is_preserved_for_task_history(self):
        agent = self.make_agent()
        content = "a" * 3906

        compacted, truncated = agent.compact_content_for_history(content)

        self.assertFalse(truncated)
        self.assertEqual(compacted, content)

    def test_default_history_content_limit_is_24000(self):
        agent = self.make_agent()

        self.assertEqual(agent.current_task_history_content_chars, 24000)

    def test_large_content_keeps_head_and_tail_with_marker(self):
        agent = self.make_agent()
        agent.current_task_history_content_chars = 1200
        content = "HEAD" + ("m" * 3000) + "TAIL"

        compacted, truncated = agent.compact_content_for_history(content)

        self.assertTrue(truncated)
        self.assertIn("HEAD", compacted)
        self.assertIn("TAIL", compacted)
        self.assertIn("middle content truncated", compacted)
        self.assertLessEqual(len(compacted), 1200)

    def test_browser_open_accepts_workspace_relative_html_file(self):
        agent = self.make_agent()
        path = agent.safe_path("index.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write("<!doctype html><title>Test</title>")

        class FakeResponse:
            status = 200

        class FakePage:
            def __init__(self):
                self.opened_url = ""

            def goto(self, url, **kwargs):
                self.opened_url = url
                return FakeResponse()

        page = FakePage()
        agent.ensure_browser_page = lambda: page
        agent.browser_snapshot = lambda: {"success": True, "url": page.opened_url}

        result = agent.browser_open("index.html")

        self.assertTrue(result["success"])
        self.assertTrue(result["opened_url"].startswith("file://"))
        self.assertIn("index.html", result["opened_url"])

    def test_browser_url_rejects_local_file_outside_workdir(self):
        agent = self.make_agent()

        with self.assertRaises(ValueError):
            agent.browser_url("file:///etc/passwd")

    def test_structured_finalizer_prompt_requires_screenshot_for_visual_checks(self):
        agent = self.make_agent()

        prompt = agent.build_structured_finalizer_prompt("check the canvas", [])

        self.assertIn("screenshot", prompt)
        self.assertIn("browser title/text/load success alone is not visual verification", prompt)

    def test_browser_screenshot_becomes_model_image_evidence(self):
        class ScreenshotEvidenceAgent(OllamaAgent):
            def __init__(self):
                super().__init__()
                self.query_image_paths = []

            def query_ollama(self, prompt, **kwargs):
                self.query_image_paths.append(list(kwargs.get("image_paths") or []))
                if len(self.query_image_paths) == 1:
                    return json.dumps({
                        "summary": "Capture screenshot",
                        "action": "browser_screenshot",
                        "filename": "shot.png",
                    })
                return json.dumps({
                    "summary": "Finish",
                    "action": "finish",
                    "message": {
                        "status": "passed",
                        "summary": "Saw screenshot evidence.",
                        "findings": [],
                        "artifacts": ["shot.png"],
                    },
                })

            def execute_action(self, action_obj):
                action = action_obj.get("action")
                if action == "browser_screenshot":
                    return {
                        "success": True,
                        "filename": str(action_obj.get("filename")),
                        "path": self.safe_path(str(action_obj.get("filename"))),
                    }
                if action == "finish":
                    return {
                        "success": True,
                        "finished": True,
                        "message": action_obj.get("message"),
                    }
                return super().execute_action(action_obj)

        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        old_workdir = os.environ.get("AGENT_WORKDIR")
        os.environ["AGENT_WORKDIR"] = tmpdir.name
        try:
            agent = ScreenshotEvidenceAgent()
        finally:
            if old_workdir is None:
                os.environ.pop("AGENT_WORKDIR", None)
            else:
                os.environ["AGENT_WORKDIR"] = old_workdir

        agent.run_agentic_task(
            "visual check",
            progress_callback=lambda event: None,
            max_iterations=2,
        )

        self.assertEqual(agent.query_image_paths[0], [])
        self.assertIn("shot.png", agent.query_image_paths[1])


if __name__ == "__main__":
    unittest.main()
