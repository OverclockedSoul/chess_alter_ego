from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import load_config
from .inference import load_inference_model, rank_moves


class MoveRequest(BaseModel):
    fen: str = Field(min_length=1)


class MoveResponse(BaseModel):
    move: str
    fen: str
    topMove: str
    selectionPolicy: str


def _split_env_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def _api_state() -> dict[str, Any]:
    config_path = Path(os.getenv("CHESS_BOT_CONFIG", "configs/restible.yaml")).resolve()
    checkpoint_path = Path(os.getenv("CHESS_BOT_CHECKPOINT", "data/models/full/best.pt")).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    config = load_config(config_path)
    model, device = load_inference_model(checkpoint_path)
    return {
        "config": config,
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "model": model,
        "selection_policy": os.getenv("CHESS_BOT_SELECTION_POLICY", "sample_probability_power"),
        "min_probability": _float_env("CHESS_BOT_MIN_PROBABILITY", 0.20),
        "below_threshold_weight_scale": _float_env("CHESS_BOT_BELOW_THRESHOLD_WEIGHT_SCALE", 0.25),
        "probability_exponent": _float_env("CHESS_BOT_PROBABILITY_EXPONENT", 2.0),
    }


app = FastAPI(title="Chess Alter Ego Bot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_split_env_list(
        "CHESS_BOT_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001",
    ),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    config_path = Path(os.getenv("CHESS_BOT_CONFIG", "configs/restible.yaml")).resolve()
    checkpoint_path = Path(os.getenv("CHESS_BOT_CHECKPOINT", "data/models/full/best.pt")).resolve()
    return {
        "status": "ok" if config_path.exists() and checkpoint_path.exists() else "degraded",
        "config": str(config_path),
        "configExists": config_path.exists(),
        "checkpoint": str(checkpoint_path),
        "checkpointExists": checkpoint_path.exists(),
        "modelLoaded": _api_state.cache_info().currsize > 0,
    }


@app.post("/move", response_model=MoveResponse)
def move(payload: MoveRequest) -> MoveResponse:
    try:
        board = chess.Board(payload.fen)
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid FEN.") from error

    if board.is_game_over() or not any(board.legal_moves):
        raise HTTPException(status_code=400, detail="Position has no legal moves.")

    try:
        state = _api_state()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    target_elo = int(state["config"]["training"]["target_self_elo"])
    inference = rank_moves(
        state["model"],
        board.fen(),
        target_elo,
        target_elo,
        selection_policy=state["selection_policy"],
        min_probability=state["min_probability"],
        below_threshold_weight_scale=state["below_threshold_weight_scale"],
        probability_exponent=state["probability_exponent"],
    )
    return MoveResponse(
        move=inference["best_move"],
        fen=board.fen(),
        topMove=inference["top_move"],
        selectionPolicy=inference["selection_policy"],
    )
