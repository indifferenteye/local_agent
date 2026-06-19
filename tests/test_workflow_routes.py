#!/usr/bin/env python3

import os
import sys
import tempfile
import types
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHATGT = os.path.join(ROOT, "chatgt")
if CHATGT not in sys.path:
    sys.path.insert(0, CHATGT)

sys.modules.setdefault("requests", types.SimpleNamespace(get=None, post=None))
IMPORT_TMPDIR = tempfile.TemporaryDirectory()
os.environ["AGENT_WORKDIR"] = IMPORT_TMPDIR.name

try:
    from flask import Flask
except ImportError:  # pragma: no cover
    Flask = None

if Flask is None:  # pragma: no cover
    raise unittest.SkipTest("Flask is not installed")

import app_state as state
from routes import register_routes


class WorkflowRouteTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.old_workdir = state.agent.working_dir
        self.old_running = state.running
        state.agent.working_dir = self.tmpdir.name
        state.running = False
        app = Flask(__name__)
        register_routes(app)
        self.client = app.test_client()

    def tearDown(self):
        state.agent.working_dir = self.old_workdir
        state.running = self.old_running
        self.tmpdir.cleanup()

    def workflow_payload(self, workflow_id="custom"):
        return {
            "id": workflow_id,
            "name": "Custom",
            "description": "A custom workflow",
            "steps": [
                {
                    "kind": "step",
                    "id": "step",
                    "type": "agentic_task",
                    "role": "planner",
                    "prompt": "Plan",
                    "condition": "",
                    "tool_guidance": "",
                    "max_iterations": 3,
                }
            ],
        }

    def test_create_get_export_delete_workflow(self):
        response = self.client.post("/workflows", json={"workflow": self.workflow_payload()})
        self.assertEqual(response.status_code, 200)

        response = self.client.get("/workflows/custom")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["workflow"]["editable"])

        response = self.client.get("/workflows/custom/export")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'"id": "custom"', response.data)

        response = self.client.delete("/workflows/custom")
        self.assertEqual(response.status_code, 200)

        response = self.client.get("/workflows/custom")
        self.assertEqual(response.status_code, 404)

    def test_clone_builtin_workflow(self):
        response = self.client.post(
            "/workflows/code_task/clone",
            json={"id": "code_clone", "name": "Code clone"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["workflow"]["id"], "code_clone")

    def test_import_rejects_invalid_workflow(self):
        payload = self.workflow_payload("bad")
        payload["steps"][0]["role"] = "invalid"

        response = self.client.post("/workflows/import", json={"workflow": payload})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid role", response.get_json()["error"])

    def test_builtin_update_is_rejected(self):
        response = self.client.put("/workflows/code_task", json={"workflow": self.workflow_payload("code_task")})

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
