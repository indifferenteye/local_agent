#!/usr/bin/env python3

"""Compatibility wrapper for the old agent_core_fixed_import module name."""

import sys

from agent_core import OllamaAgent, ProgressCallback, ProgressEvent

__all__ = ["OllamaAgent", "ProgressCallback", "ProgressEvent"]


if __name__ == "__main__":
    agent = OllamaAgent()

    if len(sys.argv) > 1:
        task_arg = " ".join(sys.argv[1:])
        print(agent.run_agentic_task(task_arg))
    else:
        agent.interactive_loop()
