FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY cybersoc_arena ./cybersoc_arena
COPY train_grpo.py ./train_grpo.py
COPY tests ./tests

RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

# Default: run a 200-episode rollout to populate runs/grpo with logs and plots.
CMD ["python", "train_grpo.py", "--steps", "200", "--seed", "42"]
