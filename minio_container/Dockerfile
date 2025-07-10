# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.4
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Create a non-privileged user.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install dependencies.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

USER appuser

COPY . .

EXPOSE 8200

# Default command to run your consumer
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port" "8200" "--reload"]
