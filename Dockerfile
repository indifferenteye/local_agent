FROM python:3.12-slim

WORKDIR /agent

RUN pip install --no-cache-dir flask requests

CMD ["python", "/agent/web-agent.py"]