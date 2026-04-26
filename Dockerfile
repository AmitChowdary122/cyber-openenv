# CyberSOC Arena -- Hugging Face Spaces / Docker image
#
# Builds the OpenEnv-compliant FastAPI server (created via
# openenv.core.env_server.web_interface.create_web_interface_app) that
# exposes the standard /reset, /step, /state, /health, /docs JSON
# endpoints + the Gradio-backed /web HumanAgent UI + the /ws WebSocket
# session at the same port the HF Space uses (7860).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app

# Install runtime deps. Training extras (torch / trl / peft) are optional --
# the env itself does not need them; they only show up in the HF Jobs script.
# gradio>=4.40 is required for the /web HumanAgent UI; if it fails to install
# the server still serves all JSON endpoints (server.py falls back gracefully).
COPY pyproject.toml requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install \
        "openenv-core>=0.2.3" \
        "fastapi" "uvicorn[standard]" "pydantic>=2" \
        "requests" "numpy" "matplotlib" \
        "gradio>=4.40" \
        "jinja2>=3.1.2"

COPY . /app
RUN pip install -e .

EXPOSE 7860 8000

# The `server` console-script (defined in pyproject.toml) launches uvicorn
# against `cybersoc_arena.server:app`, which is the OpenEnv-built FastAPI app.
CMD ["sh", "-c", "uvicorn cybersoc_arena.server:app --host ${HOST:-0.0.0.0} --port ${PORT:-7860}"]
