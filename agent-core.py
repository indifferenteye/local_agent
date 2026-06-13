#!/usr/bin/env python3
import os
import subprocess
import json
import sys
import time
from pathlib import Path
import requests
import re

class OllamaAgent:
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.working_dir = "/agent/workdir"
        os.makedirs(self.working_dir, exist_ok=True)

    def check_ollama_status(self):
        """Check if Ollama is running and available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Ollama not ready: {e}")
            return False

    def query_ollama(self, prompt, model="gemma4:e2b"):
        """Query Ollama for natural language interpretation and action planning"""
        # Wait for Ollama to be ready
        max_wait = 30
        wait_time = 0
        while not self.check_ollama_status() and wait_time < max_wait:
            print("Waiting for Ollama to start...")
            time.sleep(2)
            wait_time += 2

        if not self.check_ollama_status():
            return "Error: Ollama is not responding after waiting"

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60  # Increased timeout to 60 seconds
            )
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                print(f"Ollama API error: {response.status_code}")
                print(f"Response: {response.text}")
                return "Error communicating with Ollama"
        except requests.exceptions.Timeout:
            return "Error: Ollama request timed out (model might be too large or loading)"
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return "Error communicating with Ollama"

    def execute_command(self, command):
        """Execute a shell command and return result"""
        # Safety check - prevent dangerous commands
        dangerous_patterns = [
            r'\b(rm\s+-rf\b|\bdelete\b|\bformat\b|\bshutdown\b|\breboot\b)',
            r'\b\(.*\)\s*&&',
            r'\b\|\s*\|\s*'
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, command.lower()):
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': 'Security violation: Command contains potentially dangerous operations',
                    'returncode': 1
                }

        try:
            print(f"Executing command: {command}")
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': 'Command timed out',
                'returncode': -1
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Error executing command: {str(e)}',
                'returncode': -1
            }

    def extract_and_execute_commands(self, response_text):
        """Extract commands from response and execute them"""
        # Look for code blocks or command patterns in the response
        # Pattern 1: Markdown code blocks with shell commands
        code_block_pattern = r'```(?:shell)?\s*(.*?)```'
        matches = re.findall(code_block_pattern, response_text, re.DOTALL | re.IGNORECASE)

        # Pattern 2: Lines that start with $ or command patterns
        lines = response_text.split('\n')
        command_lines = [line.strip() for line in lines if line.strip().startswith('$') or
                        (line.strip() and not line.strip().startswith('#') and
                         not line.strip().startswith('[') and len(line.strip()) > 0)]

        # Extract commands from lines that look like they might be commands
        potential_commands = []
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('$ ') or
                (stripped and not stripped.startswith('#') and
                 not stripped.startswith('[') and len(stripped) > 2)):
                command = stripped.replace('$ ', '').strip()
                if command and not command.startswith('Error'):
                    potential_commands.append(command)

        # Prefer code blocks, then extract from text
        all_commands = matches or potential_commands

        results = []
        for cmd in all_commands:
            if cmd.strip():
                print(f"Executing extracted command: {cmd}")
                result = self.execute_command(cmd)
                results.append({
                    'command': cmd,
                    'result': result
                })

        return results

    def run_task(self, task):
        """Run a single task"""
        prompt = f"""
You are an intelligent assistant that can understand natural language and execute shell commands.
The user wants you to: {task}
Please respond with clear, concise instructions on what needs to be done.
If the task requires executing shell commands, please format them clearly in code blocks like this:
```
$ command1
$ command2
```
Only provide commands that are necessary for the task.
"""

        print(f"Processing task: {task}")
        response = self.query_ollama(prompt, model="gemma4:e2b")

        if "Error communicating with Ollama" in response:
            return response

        print("Response from Ollama:")
        print(response)

        # Try to extract and execute commands
        results = self.extract_and_execute_commands(response)

        return f"Task completed. Response: {response}\n\nCommand execution results:\n{json.dumps(results,
indent=2)}"

    def process_command(self, command):
        """Process a single command"""
        prompt = f"""
You are an intelligent assistant that can understand natural language and execute shell commands.
The user wants you to execute this specific command: {command}
Please provide clear, concise instructions on what needs to be done.
If the command is complex, break it down into steps.
If the command requires multiple shell commands, format them clearly in code blocks like this:
```
$ command1
$ command2
```
"""

        response = self.query_ollama(prompt, model="gemma4:e2b")

        if "Error communicating with Ollama" in response:
            return response

        print("Response from Ollama:")
        print(response)

        # Try to extract and execute commands
        results = self.extract_and_execute_commands(response)

        return f"Command executed. Response: {response}\n\nCommand execution results:\n{json.dumps(results,
indent=2)}"

# Test the setup
if __name__ == "__main__":
    agent = OllamaAgent()

    # First test if Ollama is working
    print("Testing Ollama connection...")
    try:
        response = agent.query_ollama("Hello", model="gemma4:e2b")
        print(f"Test response: {response[:100]}...")
    except Exception as e:
        print(f"Error testing Ollama: {e}")