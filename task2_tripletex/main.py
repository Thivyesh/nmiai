"""FastAPI entrypoint for the Tripletex accounting agent."""

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from task2_tripletex.agent import TripletexAgent
from task2_tripletex.models import SolveRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NM i AI - Tripletex Agent")
agent = TripletexAgent()


@app.post("/solve")
async def solve(request: dict):
    try:
        solve_request = SolveRequest.from_dict(request)
        logger.info("Received task: %s", solve_request.prompt[:100])

        response = await agent.solve(solve_request)

        logger.info("Task completed with status: %s", response.status)
        return JSONResponse(response.to_dict())
    except Exception as e:
        logger.exception("Error solving task")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok"}
