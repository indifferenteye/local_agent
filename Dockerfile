FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir flask requests playwright \
    && playwright install --with-deps chromium

CMD ["python", "/app/web-agent.py"]
