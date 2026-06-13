# agent-core.py
import os
import json
import subprocess
import time
import openai
from datetime import datetime

class OllamaAgent:
    def __init__(self):
        self.model = os.getenv('MODEL', 'gemma4:e2b')
        self.tools_dir = '/agent/tools'
        self.logs_dir = '/agent/logs'
        self.session_id = f"session_{int(time.time())}"

    def execute_command(self, command):
        """Execute shell commands and return results"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "success": True,
                "output": result.stdout,
                "error": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def run_tool(self, tool_name, params):
        """Run a specific tool"""
        tool_path = os.path.join(self.tools_dir, f"{tool_name}.py")

        if not os.path.exists(tool_path):
            return {"error": f"Tool {tool_name} not found"}

        try:
            # Execute tool with parameters
            cmd = f"python3 {tool_path} {json.dumps(params)}"
            result = self.execute_command(cmd)
            return result
        except Exception as e:
            return {"error": str(e)}

    def plan_and_execute(self, task):
        """Plan and execute a task using the agent"""
        print(f"[AGENT] Planning task: {task}")

        # This would typically involve calling Ollama to generate a plan
        # For demo purposes, we'll simulate this

        response = self.call_ollama(
            f"Plan how to accomplish this task: {task}. "
            "Consider using available tools like 'run_command', 'list_files', etc."
        )

        print(f"[AGENT] Plan: {response}")
        return response

    def call_ollama(self, prompt):
        """Make API calls to Ollama"""
        try:
            # This would use the actual Ollama API
            # For now, we'll simulate with a simple response
            import random
            responses = [
                "I've analyzed your request and will proceed with execution.",
                "Task breakdown complete. Starting implementation...",
                "Executing the solution as planned."
            ]
            return random.choice(responses)
        except Exception as e:
            return f"Error communicating with Ollama: {str(e)}"

    def run(self):
        """Main agent loop"""
        print(f"[AGENT] Agent started with session ID: {self.session_id}")

        while True:
            try:
                # Listen for commands or tasks
                task = input("[AGENT] Enter a task (or 'quit' to exit): ")

                if task.lower() in ['quit', 'exit']:
                    break

                if task.strip():
                    # Process the task
                    result = self.plan_and_execute(task)
                    print(f"[AGENT] Result: {result}")

            except KeyboardInterrupt:
                print("\n[AGENT] Agent shutting down...")
                break
            except Exception as e:
                print(f"[AGENT ERROR] {str(e)}")

if __name__ == "__main__":
    agent = OllamaAgent()
    agent.run()