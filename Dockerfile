FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
RUN uv pip install --system .

COPY app ./app
COPY alembic ./alembic

CMD ["python", "-m", "app.main"]

