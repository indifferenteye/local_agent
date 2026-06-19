#!/usr/bin/env python3

import os
import sys
import types
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHATGT = os.path.join(ROOT, "chatgt")
if CHATGT not in sys.path:
    sys.path.insert(0, CHATGT)

sys.modules.setdefault(
    "app_state",
    types.SimpleNamespace(messages=[], subscribers=[]),
)

from events import normalize_progress_event


class EventTests(unittest.TestCase):
    def test_normalize_progress_event_preserves_workflow_metadata(self):
        event = normalize_progress_event({
            "kind": "status",
            "iteration": 2,
            "action": "read_file",
            "workflow_step": "check",
            "workflow_step_label": "check #1",
            "workflow_loop_iteration": 1,
            "text": "Reading",
        })

        self.assertEqual(event["workflow_step"], "check")
        self.assertEqual(event["workflow_step_label"], "check #1")
        self.assertEqual(event["workflow_loop_iteration"], 1)


if __name__ == "__main__":
    unittest.main()
