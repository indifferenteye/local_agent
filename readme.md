

**Build and run the container:**
```bash
# Remove the old image
docker rmi ollama-agent

# Make sure you're in the directory with your Dockerfile
docker build -t ollama-agent .

# Run the container interactively
docker run -it --rm -p 11434:11434 ollama-agent

# In another terminal, check what files were created:
docker run -it --rm --name temp-container ollama-agent bash
# Then inside: ls -la /agent/

#example task:
Create a file test.txt