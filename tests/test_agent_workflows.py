#!/usr/bin/env python3

import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHATGT = os.path.join(ROOT, "chatgt")
if CHATGT not in sys.path:
    sys.path.insert(0, CHATGT)

from agent_workflows import (
    StepResult,
    WorkflowDefinition,
    WorkflowLoop,
    WorkflowRun,
    WorkflowRunner,
    WorkflowStep,
    clone_workflow,
    delete_custom_workflow,
    evaluate_condition,
    get_workflow,
    load_custom_workflows,
    parse_step_result,
    save_custom_workflow,
    validate_workflow,
    workflow_from_dict,
    workflow_status,
    workflow_options,
    workflow_options_for_workdir,
    workflow_to_dict,
)


class FakeAgent:
    def __init__(self):
        self.model = "default-model"
        self.model_timeout_seconds = 1
        self.run_log_level = "off"
        self.run_log_file = ".agent_runs.jsonl"
        self.run_log_detail_dir = ".agent_run_details"
        self._run_log_detail_counter = 0
        self._tmpdir = tempfile.TemporaryDirectory()
        self.routing_roles = {
            "planner": "planner-model",
            "coding": "coding-model",
            "vision": "vision-model",
        }
        self.agentic_calls = []
        self.model_calls = []
        self.check_calls = 0

    def query_ollama(self, prompt, **kwargs):
        self.model_calls.append((prompt, kwargs.get("model")))
        return '{"status": "passed", "summary": "planned", "findings": [], "artifacts": []}'

    def run_agentic_task(self, task, **kwargs):
        self.agentic_calls.append((task, kwargs.get("selected_model"), kwargs))

        if "Workflow step: check" in task or "Workflow step: verify" in task:
            self.check_calls += 1
            if self.check_calls == 1:
                return '{"status": "failed", "summary": "check failed", "findings": ["error"], "artifacts": []}'
            return '{"status": "passed", "summary": "check passed", "findings": [], "artifacts": []}'

        return '{"status": "passed", "summary": "implemented", "findings": [], "artifacts": []}'

    def safe_path(self, filename):
        return os.path.join(self._tmpdir.name, filename)

    def should_log_run_event(self, level="minimal"):
        if self.run_log_level == "off":
            return False
        if level == "full" and self.run_log_level != "full":
            return False
        return True

    def append_run_log(self, scope, event, data=None, level="minimal", run_id=None):
        if not self.should_log_run_event(level):
            return
        record = {
            "level": level,
            "scope": scope,
            "event": event,
        }
        if run_id:
            record["run_id"] = run_id
        if data:
            record["data"] = self.compact_log_data(data, level, run_id, scope, event)
        os.makedirs(os.path.dirname(self.safe_path(self.run_log_file)), exist_ok=True)
        with open(self.safe_path(self.run_log_file), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def compact_log_data(self, value, level, run_id, scope, event, path="data"):
        if isinstance(value, dict):
            return {
                key: self.compact_log_data(item, level, run_id, scope, event, f"{path}.{key}")
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self.compact_log_data(item, level, run_id, scope, event, f"{path}.{index}")
                for index, item in enumerate(value)
            ]
        if not isinstance(value, str):
            return value

        key_name = path.rsplit(".", 1)[-1]
        if level == "full" and key_name in {"prompt", "raw_output"}:
            return self.write_log_detail(value, run_id, scope, event, path)

        return value

    def write_log_detail(self, value, run_id, scope, event, path):
        self._run_log_detail_counter += 1
        run_dir = run_id or "global"
        filename = f"{self._run_log_detail_counter:04d}-{scope}-{event}-{path.replace('.', '-')}.txt"
        rel_path = os.path.join(self.run_log_detail_dir, run_dir, filename)
        detail_path = self.safe_path(rel_path)
        os.makedirs(os.path.dirname(detail_path), exist_ok=True)
        with open(detail_path, "w", encoding="utf-8") as f:
            f.write(value)
        return {
            "preview": value[:1200],
            "truncated": False,
            "chars": len(value),
            "details_ref": rel_path.replace(os.sep, "/"),
        }

    def cleanup(self):
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None


class CancelAfterFirstAgent(FakeAgent):
    def __init__(self):
        super().__init__()
        self.cancel_after_first = False

    def run_agentic_task(self, task, **kwargs):
        result = super().run_agentic_task(task, **kwargs)
        self.cancel_after_first = True
        return result


class AgentWorkflowTests(unittest.TestCase):
    def test_workflow_catalog_has_expected_v1_workflows(self):
        ids = {item["id"] for item in workflow_options()}

        self.assertIn("code_task", ids)
        self.assertIn("frontend_visual", ids)
        self.assertIn("debugging", ids)

    def test_code_task_has_visual_check_substep(self):
        workflow = get_workflow("code_task")
        self.assertIsNotNone(workflow)

        loop = next(item for item in workflow.steps if isinstance(item, WorkflowLoop))
        steps = {step.id: step for step in loop.steps}

        self.assertEqual(loop.until, 'visual_check.status == "passed"')
        self.assertEqual(steps["visual_check"].role, "vision")
        self.assertEqual(steps["visual_check"].condition, 'check.status == "passed"')
        self.assertIn("browser_screenshot", steps["visual_check"].tool_guidance)
        self.assertEqual(steps["fix"].condition, 'visual_check.status != "passed"')

    def test_condition_evaluator_supports_status_and_findings_count(self):
        results = {
            "check": StepResult("check", "failed", "failed", ["one"]),
        }

        self.assertTrue(evaluate_condition('check.status == "failed"', results))
        self.assertTrue(evaluate_condition("check.findings_count > 0", results))
        self.assertFalse(evaluate_condition('missing.status == "passed"', results))

    def test_validate_rejects_duplicate_step_ids(self):
        workflow = WorkflowDefinition(
            id="bad",
            name="Bad",
            description="",
            steps=[
                WorkflowStep("same"),
                WorkflowStep("same", "agentic_task"),
            ],
        )

        with self.assertRaises(ValueError):
            validate_workflow(workflow)

    def test_validate_rejects_unknown_condition_reference(self):
        workflow = WorkflowDefinition(
            id="bad",
            name="Bad",
            description="",
            steps=[
                WorkflowStep("step", condition='missing.status == "passed"'),
            ],
        )

        with self.assertRaises(ValueError):
            validate_workflow(workflow)

    def test_validate_rejects_invalid_role(self):
        workflow = WorkflowDefinition(
            id="bad_role",
            name="Bad role",
            description="",
            steps=[WorkflowStep("step", role="bad")],
        )

        with self.assertRaises(ValueError):
            validate_workflow(workflow)

    def test_validate_rejects_reserved_builtin_id(self):
        workflow = WorkflowDefinition(
            id="code_task",
            name="Bad",
            description="",
            steps=[WorkflowStep("step")],
        )

        with self.assertRaises(ValueError):
            validate_workflow(workflow, reserved_ids={"code_task"})

    def test_workflow_serialization_round_trips_loop(self):
        workflow = WorkflowDefinition(
            id="custom",
            name="Custom",
            description="",
            steps=[
                WorkflowLoop(
                    "loop",
                    until='check.status == "passed"',
                    max_iterations=2,
                    steps=[WorkflowStep("check", role="coding")],
                )
            ],
        )

        restored = workflow_from_dict(workflow_to_dict(workflow))

        self.assertEqual(restored.id, "custom")
        self.assertIsInstance(restored.steps[0], WorkflowLoop)
        self.assertEqual(restored.steps[0].steps[0].role, "coding")

    def test_custom_workflow_persistence_and_options(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(load_custom_workflows(tmpdir), [])
            workflow = WorkflowDefinition(
                id="custom",
                name="Custom",
                description="Saved workflow",
                steps=[WorkflowStep("step", role="planner")],
            )

            save_custom_workflow(tmpdir, workflow)
            loaded = load_custom_workflows(tmpdir)
            options = workflow_options_for_workdir(tmpdir)

            self.assertEqual(loaded[0].id, "custom")
            self.assertTrue(any(option["id"] == "custom" and option["editable"] for option in options))
            self.assertIsNotNone(get_workflow("custom", tmpdir))

    def test_invalid_custom_workflow_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, ".agent_workflows.json"), "w", encoding="utf-8") as f:
                f.write("{not json")

            self.assertEqual(load_custom_workflows(tmpdir), [])

    def test_clone_and_delete_custom_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cloned = clone_workflow(tmpdir, "code_task", new_id="code_clone", new_name="Code clone")

            self.assertEqual(cloned.id, "code_clone")
            self.assertIsNotNone(get_workflow("code_clone", tmpdir))
            self.assertTrue(delete_custom_workflow(tmpdir, "code_clone"))
            self.assertIsNone(get_workflow("code_clone", tmpdir))

    def test_non_json_step_result_needs_changes_by_default(self):
        result = parse_step_result("check", "Contents of a file without structured status")

        self.assertEqual(result.status, "needs_changes")
        self.assertIn("structured JSON", result.findings[0])

    def test_needed_changes_status_is_normalized(self):
        result = parse_step_result(
            "plan",
            '{"status": "needed_changes", "summary": "needs work", "findings": [], "artifacts": []}',
        )

        self.assertEqual(result.status, "needs_changes")

    def test_passing_verification_can_end_with_warnings(self):
        run = WorkflowRun("do work", "test")
        run.results = {
            "implement": StepResult("implement", "failed", "implementation struggled"),
            "check": StepResult("check", "passed", "verified"),
        }

        self.assertEqual(workflow_status(run), "passed_with_warnings")

    def test_passing_verification_clears_advisory_plan_warning(self):
        run = WorkflowRun("do work", "test")
        run.results = {
            "plan": StepResult("plan", "needs_changes", "plan requested changes"),
            "implement": StepResult("implement", "passed", "implemented"),
            "check": StepResult("check", "passed", "verified"),
        }

        self.assertEqual(workflow_status(run), "passed")

    def test_workflow_prompt_uses_finish_action_contract(self):
        workflow = WorkflowDefinition(
            id="test",
            name="Test",
            description="",
            steps=[WorkflowStep("plan", role="planner", prompt="Plan")],
        )
        agent = FakeAgent()
        self.addCleanup(agent.cleanup)
        runner = WorkflowRunner(agent)

        prompt = runner.build_step_prompt(workflow.steps[0], WorkflowRun("do work", "test"), "do work")

        self.assertIn('"action": "finish"', prompt)
        self.assertIn("finish.message", prompt)
        self.assertNotIn("Return useful work for this step", prompt)

    def test_runner_retries_until_check_passes(self):
        workflow = WorkflowDefinition(
            id="test",
            name="Test",
            description="",
            steps=[
                WorkflowStep("plan", role="planner", prompt="Plan"),
                WorkflowLoop(
                    id="loop",
                    max_iterations=3,
                    until='check.status == "passed"',
                    steps=[
                        WorkflowStep("check", "agentic_task", role="coding", prompt="Check"),
                        WorkflowStep("fix", "agentic_task", role="coding", condition='check.status != "passed"', prompt="Fix"),
                    ],
                ),
            ],
        )
        agent = FakeAgent()
        self.addCleanup(agent.cleanup)
        runner = WorkflowRunner(agent)

        summary = runner.run(workflow, "do work")

        self.assertIn("passed", summary)
        self.assertEqual(agent.check_calls, 2)
        self.assertEqual(agent.agentic_calls[0][1], "planner-model")
        self.assertTrue(any(model == "coding-model" for _, model, _ in agent.agentic_calls))
        self.assertTrue(all(call_kwargs.get("require_structured_result") for _, _, call_kwargs in agent.agentic_calls))
        self.assertEqual(agent.agentic_calls[0][2].get("max_iterations"), 8)

    def test_minimal_workflow_log_records_input_and_output(self):
        workflow = WorkflowDefinition(
            id="test",
            name="Test",
            description="",
            steps=[WorkflowStep("plan", role="planner", prompt="Plan")],
        )
        agent = FakeAgent()
        self.addCleanup(agent.cleanup)
        agent.run_log_level = "minimal"
        runner = WorkflowRunner(agent)

        runner.run(workflow, "do work")

        log_path = agent.safe_path(".agent_runs.jsonl")
        self.assertTrue(os.path.exists(log_path))
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn('"input": "do work"', content)
        self.assertIn('"scope": "workflow"', content)
        self.assertIn('"event": "workflow_summary"', content)
        self.assertIn('"summary"', content)

    def test_full_workflow_log_records_internal_events(self):
        workflow = WorkflowDefinition(
            id="test",
            name="Test",
            description="",
            steps=[WorkflowStep("plan", role="planner", prompt="Plan")],
        )
        agent = FakeAgent()
        self.addCleanup(agent.cleanup)
        agent.run_log_level = "full"
        runner = WorkflowRunner(agent)

        runner.run(workflow, "do work")

        log_path = agent.safe_path(".agent_runs.jsonl")
        self.assertTrue(os.path.exists(log_path))
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn('"event": "workflow_started"', content)
        self.assertIn('"event": "step_started"', content)
        self.assertIn('"details_ref"', content)

        detail_root = agent.safe_path(".agent_run_details")
        self.assertTrue(os.path.isdir(detail_root))
        detail_text = ""
        for root, _, files in os.walk(detail_root):
            for filename in files:
                with open(os.path.join(root, filename), "r", encoding="utf-8") as f:
                    detail_text += f.read()
        self.assertIn("Workflow step: plan", detail_text)
        self.assertIn('"summary": "implemented"', detail_text)

    def test_step_iteration_budget_is_passed_to_agent(self):
        workflow = WorkflowDefinition(
            id="test",
            name="Test",
            description="",
            steps=[WorkflowStep("plan", role="planner", prompt="Plan", max_iterations=3)],
        )
        agent = FakeAgent()
        self.addCleanup(agent.cleanup)
        runner = WorkflowRunner(agent)

        runner.run(workflow, "do work")

        self.assertEqual(agent.agentic_calls[0][2].get("max_iterations"), 3)
        self.assertTrue(agent.agentic_calls[0][2].get("require_structured_result"))

    def test_workflow_stops_before_next_step_when_cancelled(self):
        workflow = WorkflowDefinition(
            id="test",
            name="Test",
            description="",
            steps=[
                WorkflowStep("first", role="coding", prompt="First"),
                WorkflowStep("second", role="coding", prompt="Second"),
            ],
        )
        agent = CancelAfterFirstAgent()
        self.addCleanup(agent.cleanup)
        runner = WorkflowRunner(agent)

        summary = runner.run(
            workflow,
            "do work",
            cancel_checker=lambda: agent.cancel_after_first,
        )

        self.assertIn("cancelled", summary)
        self.assertEqual(len(agent.agentic_calls), 1)

    def test_step_progress_callback_adds_workflow_step_metadata(self):
        agent = FakeAgent()
        self.addCleanup(agent.cleanup)
        runner = WorkflowRunner(agent)
        events = []
        callback = runner.step_progress_callback(
            events.append,
            WorkflowStep("check", role="coding"),
            loop_iteration=2,
        )

        callback({"kind": "status", "iteration": 1, "text": "Working"})

        self.assertEqual(events[0]["workflow_step"], "check")
        self.assertEqual(events[0]["workflow_step_label"], "check #2")
        self.assertEqual(events[0]["workflow_loop_iteration"], 2)


if __name__ == "__main__":
    unittest.main()
