# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.4
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/app/.cache/matplotlib

# --- NEW: Install gosu for privilege dropping ---
# We run as root to install packages
RUN apt-get update && apt-get install -y --no-install-recommends gosu && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a non-privileged user.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/bin/bash" \
    --uid "${UID}" \
    appuser

# Install dependencies.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Copy the application code.
COPY . .

# Pre-create the cache directory and set ownership.
RUN mkdir -p /app/.cache/matplotlib && chown -R appuser:appuser /app

# Add and configure the entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]

# --- REMOVED: The USER instruction is no longer needed here ---
# USER appuser

# Expose the port.
EXPOSE 8000

# This command is passed to the entrypoint script.
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]