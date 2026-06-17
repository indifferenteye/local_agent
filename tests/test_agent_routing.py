#!/usr/bin/env python3

import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHATGT = os.path.join(ROOT, "chatgt")
if CHATGT not in sys.path:
    sys.path.insert(0, CHATGT)

from agent_routing import deterministic_classify, route_task


class AgentRoutingTests(unittest.TestCase):
    def test_code_task_routes_to_coding_role(self):
        decision = route_task(
            "fix this React component",
            default_model="default-model",
            roles={"coding": "code-model"},
        )

        self.assertEqual(decision.classification.task_type, "code_edit")
        self.assertEqual(decision.selected_role, "coding")
        self.assertEqual(decision.selected_model, "code-model")

    def test_image_generation_is_classified_but_uses_default_local_model(self):
        decision = route_task(
            "generate an image of a castle",
            default_model="default-model",
            roles={"coding": "code-model", "vision": "vision-model"},
        )

        self.assertEqual(decision.classification.task_type, "image_generation")
        self.assertEqual(decision.selected_role, "default")
        self.assertEqual(decision.selected_model, "default-model")
        self.assertIn("not configured", decision.reason)

    def test_uploaded_image_routes_to_vision_role(self):
        decision = route_task(
            "make it brighter",
            default_model="default-model",
            roles={"vision": "vision-model"},
            task_images=["uploads/photo.png"],
        )

        self.assertEqual(decision.classification.task_type, "image_editing")
        self.assertEqual(decision.selected_role, "vision")
        self.assertEqual(decision.selected_model, "vision-model")

    def test_translation_uses_default_in_balanced_mode(self):
        decision = route_task(
            "translate this to German",
            default_model="default-model",
            roles={"router": "router-model"},
            quality_mode="balanced",
        )

        self.assertEqual(decision.classification.task_type, "translation")
        self.assertEqual(decision.selected_role, "default")
        self.assertEqual(decision.selected_model, "default-model")

    def test_landing_page_with_hero_image_is_primary_code_task(self):
        classification = deterministic_classify("make a landing page with a hero image")

        self.assertEqual(classification.task_type, "code_edit")
        self.assertIn("image_generation", classification.secondary_task_types)
        self.assertTrue(classification.needs_tools)

    def test_unset_role_falls_back_to_default_model(self):
        decision = route_task(
            "fix this Python bug",
            default_model="default-model",
            roles={},
            quality_mode="balanced",
        )

        self.assertEqual(decision.selected_role, "coding")
        self.assertEqual(decision.selected_model, "default-model")
        self.assertTrue(decision.fallback_used)

    def test_balanced_mode_escalates_ambiguous_tool_code_classifier_result(self):
        def classifier(prompt, model):
            return """
            {
              "task_type": "debugging",
              "secondary_task_types": [],
              "requested_artifact": "fixed_behavior",
              "required_capabilities": ["code_reasoning", "tool_use"],
              "risk": "medium",
              "complexity": "medium",
              "confidence": 0.83,
              "needs_images": false,
              "needs_tools": true
            }
            """

        decision = route_task(
            "it does not work anymore",
            default_model="default-model",
            roles={"router": "router-model", "coding": "code-model"},
            quality_mode="balanced",
            classifier=classifier,
        )

        self.assertEqual(decision.classification_source, "model")
        self.assertEqual(decision.selected_role, "coding")
        self.assertEqual(decision.selected_model, "code-model")

    def test_economy_mode_still_escalates_code_tasks(self):
        decision = route_task(
            "run tests and fix the failure",
            default_model="default-model",
            roles={"router": "router-model", "coding": "code-model"},
            quality_mode="economy",
        )

        self.assertEqual(decision.selected_role, "coding")
        self.assertEqual(decision.selected_model, "code-model")

    def test_obvious_plain_chat_does_not_call_classifier(self):
        calls = []

        def classifier(prompt, model):
            calls.append((prompt, model))
            return "{}"

        decision = route_task(
            "hello",
            default_model="default-model",
            roles={"router": "router-model"},
            classifier=classifier,
        )

        self.assertEqual(decision.selected_role, "default")
        self.assertEqual(calls, [])

    def test_debug_details_include_classifier_and_final_decision(self):
        def classifier(prompt, model):
            return """
            {
              "task_type": "planning",
              "secondary_task_types": [],
              "requested_artifact": "plan",
              "required_capabilities": ["reasoning"],
              "risk": "medium",
              "complexity": "medium",
              "confidence": 0.86,
              "needs_images": false,
              "needs_tools": false
            }
            """

        decision = route_task(
            "make it better",
            default_model="default-model",
            roles={"router": "router-model", "planner": "planner-model"},
            classifier=classifier,
        )

        self.assertEqual(decision.classification_source, "model")
        self.assertTrue(decision.debug_details["classifier_attempted"])
        self.assertEqual(decision.debug_details["classifier_model"], "router-model")
        self.assertEqual(decision.debug_details["selected_role"], "planner")
        self.assertEqual(decision.debug_details["selected_model"], "planner-model")
        self.assertIn("deterministic_classification", decision.debug_details)
        self.assertIn("model_classification", decision.debug_details)


if __name__ == "__main__":
    unittest.main()
