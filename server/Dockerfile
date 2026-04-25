# OpenEnv-template Dockerfile location: server/Dockerfile.
# The actual image build instructions are kept in the repo-root Dockerfile;
# we duplicate them here because some validators / docker build commands
# expect to find a Dockerfile inside server/.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app

COPY pyproject.toml requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install \
        "openenv-core>=0.2.3" \
        "fastapi" "uvicorn[standard]" "pydantic>=2" \
        "requests" "numpy" "matplotlib"

COPY . /app
RUN pip install -e .

EXPOSE 7860 8000

CMD ["sh", "-c", "uvicorn cybersoc_arena.server:app --host ${HOST:-0.0.0.0} --port ${PORT:-7860}"]
