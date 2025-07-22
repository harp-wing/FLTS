# syntax=docker/dockerfile:1

FROM continuumio/miniconda3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install conda env from environment.yaml (mounted)
RUN --mount=type=bind,source=environment.yaml,target=environment.yaml \
    conda env create -f environment.yaml

# Set default shell to conda run with the new environment
SHELL ["conda", "run", "-n", "appenv", "/bin/bash", "-c"]

# Switch to non-root user
USER appuser

# Copy app code
COPY . .

# Expose the app port
EXPOSE 8002

# Run the app
CMD ["conda", "run", "--no-capture-output", "-n", "appenv", "python", "main.py"]