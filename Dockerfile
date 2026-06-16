FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir flask requests

CMD ["python", "/app/web-agent.py"]
