from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def render_http_lichess_bot_config(
    output_path: str | Path,
    *,
    project_root: str | Path | None = None,
    token: str | None = None,
    allow_missing_token: bool = False,
) -> Path:
    root = Path(project_root or Path.cwd()).resolve()
    rendered_output_path = Path(output_path).resolve()
    rendered_output_path.parent.mkdir(parents=True, exist_ok=True)

    token_value = token or os.getenv("LICHESS_BOT_TOKEN")
    if not token_value and not allow_missing_token:
        raise RuntimeError("LICHESS_BOT_TOKEN is required to render the lichess-bot runner config.")

    payload: dict[str, Any] = {
        "token": token_value or "SET_ME",
        "url": os.getenv("LICHESS_BASE_URL", "https://lichess.org").rstrip("/") + "/",
        "engine": {
            "dir": str(root),
            "name": str(Path("src") / "restible_bot" / "http_uci_engine.py"),
            "interpreter": sys.executable,
            "interpreter_options": [],
            "working_dir": str(root),
            "protocol": "uci",
            "ponder": False,
            "engine_options": {},
            "uci_options": {
                "Move Overhead": 200,
            },
            "draw_or_resign": {
                "offer_draw_enabled": False,
                "resign_enabled": False,
            },
        },
        "challenge": {
            "concurrency": 3,
            "sort_by": "first",
            "preference": "none",
            "accept_bot": True,
            "only_bot": False,
            "max_increment": 180,
            "min_increment": 0,
            "max_base": 1800,
            "min_base": 0,
            "variants": ["standard"],
            "time_controls": ["bullet", "blitz", "rapid", "classical"],
            "modes": ["casual"],
            "max_simultaneous_games_per_user": 3,
        },
        "matchmaking": {
            "allow_matchmaking": False,
            "allow_during_games": False,
            "challenge_timeout": 1,
            "challenge_initial_time": [300],
            "challenge_increment": [0],
            "challenge_mode": "casual",
            "challenge_variant": "standard",
            "rating_preference": "none",
        },
        "abort_time": 30,
        "fake_think_time": False,
        "rate_limiting_delay": 0,
        "move_overhead": 2000,
        "max_takebacks_accepted": 0,
        "quit_after_all_games_finish": False,
        "correspondence": {
            "move_time": 60,
            "checkin_period": 300,
            "disconnect_time": 150,
            "ponder": False,
        },
        "greeting": {
            "hello": "",
            "goodbye": "",
            "hello_spectators": "",
            "goodbye_spectators": "",
        },
    }
    rendered_output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return rendered_output_path


def main() -> None:
    parser = argparse.ArgumentParser(prog="render-http-lichess-config")
    parser.add_argument("--output", default="data/artifacts/lichess-bot/http-config.yml")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-missing-token", action="store_true")
    args = parser.parse_args()

    output = render_http_lichess_bot_config(
        args.output,
        project_root=args.project_root,
        allow_missing_token=args.allow_missing_token,
    )
    print(output)


if __name__ == "__main__":
    main()
