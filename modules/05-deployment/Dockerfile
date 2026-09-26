FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml uv.lock* ./
COPY src ./src
RUN pip install --no-cache-dir uv && uv sync --extra deployment --no-dev
COPY modules/05-deployment/service ./service

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
