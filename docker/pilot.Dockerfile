# Test image for the pilot episodes (boltons): Python 3.12 with the exact pilot test environment.
#
#   docker build -f docker/pilot.Dockerfile -t testorcode/pilot-py312:pytest-8.4.2 .
#   SANDBOX_ISOLATION=docker SANDBOX_DOCKER_IMAGE=testorcode/pilot-py312:pytest-8.4.2 python scripts/run_experiment.py ...
#
# The sandbox (src/testorcode/tools/sandbox.py) runs it with no network, a read-only root filesystem, all capabilities
# dropped, the caller's uid, resource limits and the episode worktree mounted read-only at /work.  Nothing from the
# repository under test is baked in: the worktree is mounted per run.  Rebuild after changing
# scripts/pilot/requirements-pilot.txt, and record the image digest (`docker images --digests`) with the results.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY scripts/pilot/requirements-pilot.txt /tmp/requirements-pilot.txt
RUN python -m pip install --no-deps -r /tmp/requirements-pilot.txt \
    && python -m pip check \
    && rm /tmp/requirements-pilot.txt

WORKDIR /work
