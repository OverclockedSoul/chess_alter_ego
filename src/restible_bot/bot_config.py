from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import chess
import chess.engine
import yaml

from .config import project_root, resolve_path
from .utils import ensure_parent


def default_checkpoint_path(config: dict[str, Any]) -> Path:
    root = project_root(config)
    full_checkpoint = root / "data" / "models" / "full" / "best.pt"
    smoke_checkpoint = root / "data" / "models" / "smoke" / "best.pt"
    if full_checkpoint.exists():
        return full_checkpoint
    if smoke_checkpoint.exists():
        return smoke_checkpoint
    raise FileNotFoundError("No checkpoint found. Train the model first or pass --checkpoint explicitly.")


def render_lichess_bot_config(config: dict[str, Any], checkpoint_path: Path | None = None) -> Path:
    root = project_root(config)
    checkpoint_path = (checkpoint_path or default_checkpoint_path(config)).resolve()
    output_path = resolve_path(config, config["bot"]["config_output"])
    ensure_parent(output_path)

    token = os.getenv(config["bot"]["token_env"], "SET_ME")
    payload: dict[str, Any] = {
        "token": token,
        "url": config["lichess"]["base_url"].rstrip("/") + "/",
        "engine": {
            "dir": str(root),
            "name": str(Path("src") / "restible_bot" / "uci_engine.py"),
            "interpreter": sys.executable,
            "interpreter_options": [],
            "working_dir": str(root),
            "protocol": "uci",
            "ponder": False,
            "engine_options": {
                "config": config["__config_path__"],
                "checkpoint": str(checkpoint_path),
            },
            "uci_options": {
                "Threads": 1,
                "Move Overhead": 200,
            },
            "draw_or_resign": {
                "offer_draw_enabled": False,
                "resign_enabled": False,
            },
        },
        "challenge": {
            "concurrency": 1,
            "variants": ["standard"],
            "time_controls": ["rapid"],
            "modes": ["casual"],
            "accept_bot": True,
            "only_bot": False,
        },
        "matchmaking": {
            "allow_matchmaking": False,
        },
    }
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def verify_local_bot_setup(config: dict[str, Any], checkpoint_path: Path | None = None) -> dict[str, Any]:
    checkpoint = (checkpoint_path or default_checkpoint_path(config)).resolve()
    config_path = render_lichess_bot_config(config, checkpoint)
    root = project_root(config)
    engine_command = [
        sys.executable,
        str(root / "src" / "restible_bot" / "uci_engine.py"),
        f"--config={config['__config_path__']}",
        f"--checkpoint={checkpoint}",
    ]
    engine = chess.engine.SimpleEngine.popen_uci(engine_command, cwd=root)
    try:
        board = chess.Board()
        result = engine.play(board, chess.engine.Limit(time=0.01))
    finally:
        engine.quit()
    return {
        "config_path": str(config_path),
        "checkpoint": str(checkpoint),
        "token_present": bool(os.getenv(config["bot"]["token_env"])),
        "predicted_move": result.move.uci(),
    }


def run_lichess_bot(config: dict[str, Any], checkpoint_path: Path | None = None) -> None:
    if not os.getenv(config["bot"]["token_env"]):
        raise RuntimeError(f"Missing {config['bot']['token_env']} in the environment or .env file.")
    config_path = render_lichess_bot_config(config, checkpoint_path)
    root = project_root(config)
    subprocess.run(
        [sys.executable, str(root / "third_party" / "lichess-bot" / "lichess-bot.py"), "--config", str(config_path)],
        check=True,
        cwd=root,
    )

