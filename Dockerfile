FROM python:3.9-slim

# Install system dependencies including zstd for Ollama
RUN apt-get update && apt-get install -y \
    curl \
    zstd \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama using the official installation method
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create working directories
RUN mkdir -p /agent/tools /agent/workdir
WORKDIR /agent

# Copy Python script
COPY agent-core.py .

# Expose Ollama port
EXPOSE 11434

# Install Python dependencies with --break-system-packages flag
RUN pip install --break-system-packages requests

# Create a startup script that handles the model pulling
RUN echo '#!/bin/bash\n\
ollama serve &\n\
sleep 5\n\
ollama pull gemma4:e2b\n\
python3 agent-core.py\n' > start.sh && chmod +x start.sh

# Start with our custom startup script
CMD ["/agent/start.sh"]