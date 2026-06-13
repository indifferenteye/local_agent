@echo off
echo Building Ollama agent container...

REM Build the Docker image
docker build -t ollama-agent .

if %ERRORLEVEL% EQU 0 (
    echo Container built successfully!
    echo Running container with agent capabilities...

    REM Run the container
    docker run -it --rm ^
      -p 11434:11434 ^
      --name ollama-agent-container ^
      ollama-agent

    echo Container stopped.
) else (
    echo Failed to build container.
)

pause