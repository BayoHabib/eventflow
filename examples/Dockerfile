# syntax=docker/dockerfile:1
# Examples-only image for EventFlow.
#
# This image installs the library from the built wheel (+ dev extras)
# and runs the smoke example. It is NOT a prediction API and NOT a
# deployment service: EventFlow prepares ML-ready features; models and
# serving live downstream.
#
# Build from the repository root, AFTER `python -m build` (the wheel
# must exist under dist/):
#   docker build -f examples/Dockerfile -t eventflow-examples:0.1.0 .
#   docker run --rm eventflow-examples:0.1.0
#
# Status: NOT built in the release-verification VM (no Docker daemon
# there); the equivalent steps were verified in a fresh venv instead
# (see examples/README.md). Build it on a machine with Docker before
# relying on it.
FROM python:3.12-slim

WORKDIR /app

# The built wheel (0.1.0) plus the example and its recipe config.
# Configs are user-supplied data, not shipped in the wheel, so they are
# copied in explicitly for the example run.
COPY dist/eventflow-0.1.0-py3-none-any.whl /app/
COPY examples/ /app/examples/
COPY configs/ /app/configs/

# Runtime deps come from the wheel; [dev] adds the test/lint toolchain
# so the image can also run the suite's example test.
RUN pip install --no-cache-dir "/app/eventflow-0.1.0-py3-none-any.whl[dev]" \
    && eventflow --help \
    && eventflow validate --config /app/configs/recipes/chicago_crime_v1.yaml

CMD ["python", "examples/chicago_quickstart.py"]
