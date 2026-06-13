# agent-entrypoint.bat (Windows batch file)
@echo off
echo Starting Ollama agent system...

REM Start Ollama in background
start /b ollama serve

REM Wait for Ollama to be ready
timeout /t 5 /nobreak >nul

REM Pull the required model
echo Pulling model: %MODEL%
ollama pull %MODEL%

REM Initialize agent components
echo Initializing agent system...
python3 /agent/agent-core.py

REM Keep container alive
cmd /k "echo Container running... Press Ctrl+C to stop"