# api/main.py
from fastapi import FastAPI, Request
from app.env import CyberEnv
from app.models import Action
from app.grader import grade_state
from app.tasks import TASKS
import uvicorn

app = FastAPI(title="CyberEnv API")
env = CyberEnv()


@app.post("/reset")
async def reset_endpoint(request: Request):
    """
    Reset environment.
    Optional JSON body:
    {
        "task_id": "easy" | "medium" | "hard"
    }
    """
    try:
        body = await request.json()
    except:
        body = {}

    task_id = body.get("task_id")

    # detect task difficulty
    if task_id:
        difficulty = task_id
    else:
        difficulty = "easy"

    env = CyberEnv(difficulty=difficulty)
    obs = env.reset()

    # Override task if provided
    if task_id:
        for t in TASKS:
            if t["id"] == task_id:
                env.current_task = t
                break

    return {
        "observation": obs.dict(),
        "state": env.state(),
        "task": env.current_task
    }


@app.post("/step")
async def step_endpoint(action: Action):
    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.dict(),
        "reward": reward.value,
        "done": done,
        "info": info,
        "state": env.state()
    }


@app.get("/state")
async def state_endpoint():
    return {"state": env.state()}


@app.post("/grade")
async def grade_endpoint():
    score = grade_state(env.state())
    return {"score": score}


@app.get("/grade")
async def grade_get_endpoint():
    score = grade_state(env.state())
    return {"score": score}


@app.get("/tasks")
async def list_tasks():
    return TASKS

@app.get("/")
async def root():
    return {"status": "CyberEnv API is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)