from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import signal

import chess
import chess.engine
import requests
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


def _auth_headers(config: dict[str, Any]) -> dict[str, str]:
    token = os.getenv(config["bot"]["token_env"])
    if not token:
        raise RuntimeError(f"Missing {config['bot']['token_env']} in the environment or .env file.")
    return {"Authorization": f"Bearer {token}"}


def fetch_account_state(config: dict[str, Any]) -> dict[str, Any]:
    headers = _auth_headers(config)
    base_url = config["lichess"]["base_url"].rstrip("/")
    account = requests.get(f"{base_url}/api/account", headers=headers, timeout=30)
    account.raise_for_status()
    playing = requests.get(f"{base_url}/api/account/playing", headers=headers, timeout=30)
    playing.raise_for_status()
    return {
        "account": account.json(),
        "playing": playing.json(),
    }


def render_lichess_bot_config(
    config: dict[str, Any],
    checkpoint_path: Path | None = None,
    *,
    output_path: str | Path | None = None,
    allow_matchmaking: bool = False,
    rated: bool = False,
    initial_time: int = 480,
    increment: int = 0,
    concurrency: int = 1,
    allow_during_games: bool = False,
    opponent_min_rating: int | None = None,
    opponent_max_rating: int | None = None,
    opponent_rating_difference: int | None = 500,
    accept_bot: bool = True,
    accept_casual_challenges: bool = False,
    selection_policy: str = "sample_probability_power",
    min_probability: float = 0.20,
    below_threshold_weight_scale: float = 0.25,
    probability_exponent: float = 2.0,
) -> Path:
    root = project_root(config)
    checkpoint_path = (checkpoint_path or default_checkpoint_path(config)).resolve()
    rendered_output_path = resolve_path(config, output_path or config["bot"]["config_output"])
    ensure_parent(rendered_output_path)

    token = os.getenv(config["bot"]["token_env"], "SET_ME")
    mode = "rated" if rated else "casual"
    challenge_modes = ["rated", "casual"] if rated and accept_casual_challenges else [mode]
    matchmaking: dict[str, Any] = {
        "allow_matchmaking": allow_matchmaking,
        "allow_during_games": allow_during_games,
        "challenge_timeout": 1,
        "challenge_initial_time": [initial_time],
        "challenge_increment": [increment],
        "challenge_mode": mode,
        "rating_preference": "none",
    }
    if opponent_min_rating is not None:
        matchmaking["opponent_min_rating"] = opponent_min_rating
    if opponent_max_rating is not None:
        matchmaking["opponent_max_rating"] = opponent_max_rating
    if opponent_rating_difference is not None:
        matchmaking["opponent_rating_difference"] = opponent_rating_difference

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
                "selection-policy": selection_policy,
                "min-probability": str(min_probability),
                "below-threshold-weight-scale": str(below_threshold_weight_scale),
                "probability-exponent": str(probability_exponent),
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
            "concurrency": concurrency,
            "variants": ["standard"],
            "time_controls": ["bullet", "blitz", "rapid"],
            "modes": challenge_modes,
            "accept_bot": accept_bot,
            "only_bot": not accept_bot,
        },
        "matchmaking": matchmaking,
    }
    rendered_output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return rendered_output_path


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


def _rapid_perf(account: dict[str, Any]) -> dict[str, Any]:
    return (account.get("perfs") or {}).get("rapid") or {}


def _active_game_count(playing_payload: dict[str, Any]) -> int:
    now_playing = playing_payload.get("nowPlaying")
    if isinstance(now_playing, list):
        return len(now_playing)
    return int(playing_payload.get("playing", 0) or 0)


def _fetch_game_export(config: dict[str, Any], game_id: str) -> dict[str, Any] | None:
    base_url = config["lichess"]["base_url"].rstrip("/")
    response = requests.get(
        f"{base_url}/game/export/{game_id}",
        headers={"Accept": "application/json"},
        params={"moves": "false", "pgnInJson": "true"},
        timeout=30,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    if not response.text.strip():
        return None
    return json.loads(response.text)


def _create_direct_challenge(
    config: dict[str, Any],
    username: str,
    *,
    rated: bool,
    initial_time: int,
    increment: int,
) -> dict[str, Any]:
    headers = _auth_headers(config)
    base_url = config["lichess"]["base_url"].rstrip("/")
    response = requests.post(
        f"{base_url}/api/challenge/{username}",
        headers=headers,
        data={
            "rated": str(rated).lower(),
            "variant": "standard",
            "clock.limit": initial_time,
            "clock.increment": increment,
        },
        timeout=30,
    )
    body = response.json()
    if response.status_code >= 400:
        error_message = body.get("error") or body.get("global") or str(body)
        raise RuntimeError(f"Challenge request to {username} failed: {error_message}")
    return body


def _run_direct_challenge_series(
    process: subprocess.Popen[str],
    config: dict[str, Any],
    *,
    username: str,
    rated: bool,
    initial_time: int,
    increment: int,
    max_games: int,
    log_path: Path,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    base_url = config["lichess"]["base_url"].rstrip("/")
    headers = _auth_headers(config)
    challenge_wait_seconds = max(120, initial_time * 4)
    game_wait_seconds = max(900, initial_time * 8)

    account_username = fetch_account_state(config)["account"]["username"]

    def _score_from_game(game: dict[str, Any]) -> float:
        players = game["players"]
        white_name = players["white"]["user"]["name"]
        black_name = players["black"]["user"]["name"]
        winner = game.get("winner")
        if winner is None:
            return 0.5
        if winner == "white":
            return 1.0 if white_name == account_username else 0.0
        return 1.0 if black_name == account_username else 0.0

    for _ in range(max_games):
        if process.poll() is not None:
            raise RuntimeError(f"lichess-bot exited early with code {process.returncode}. See {log_path}.")

        challenge = _create_direct_challenge(
            config,
            username,
            rated=rated,
            initial_time=initial_time,
            increment=increment,
        )
        challenge_id = challenge.get("id")
        if not challenge_id:
            raise RuntimeError(f"Failed to create challenge for {username}: {challenge}")

        challenge_deadline = time.time() + challenge_wait_seconds
        while time.time() < challenge_deadline:
            if process.poll() is not None:
                raise RuntimeError(f"lichess-bot exited early with code {process.returncode}. See {log_path}.")
            game = _fetch_game_export(config, challenge_id)
            if game and game.get("status") != "created":
                break
            time.sleep(5)
        else:
            requests.post(f"{base_url}/api/challenge/{challenge_id}/cancel", headers=headers, timeout=30)
            raise RuntimeError(f"Challenge {challenge_id} was not accepted in time. See {log_path}.")

        game_deadline = time.time() + game_wait_seconds
        while time.time() < game_deadline:
            if process.poll() is not None:
                raise RuntimeError(f"lichess-bot exited early with code {process.returncode}. See {log_path}.")
            game = _fetch_game_export(config, challenge_id)
            if game and game.get("status") not in {"created", "started"}:
                score = _score_from_game(game)
                results.append(
                    {
                        "game_id": challenge_id,
                        "score": score,
                        "status": game.get("status"),
                        "winner": game.get("winner"),
                    }
                )
                break
            time.sleep(5)
        else:
            raise RuntimeError(f"Game {challenge_id} did not complete in time. See {log_path}.")

    wins = sum(1 for item in results if item["score"] == 1.0)
    draws = sum(1 for item in results if item["score"] == 0.5)
    losses = sum(1 for item in results if item["score"] == 0.0)
    return {
        "username": account_username,
        "opponent": username,
        "games_completed": len(results),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": sum(item["score"] for item in results),
        "log_path": str(log_path),
    }


def run_lichess_bot(
    config: dict[str, Any],
    checkpoint_path: Path | None = None,
    *,
    max_games: int | None = None,
    max_runtime_seconds: int | None = None,
    allow_matchmaking: bool = False,
    rated: bool = False,
    initial_time: int = 480,
    increment: int = 0,
    concurrency: int = 1,
    poll_interval_seconds: int = 30,
    opponent_min_rating: int | None = None,
    opponent_max_rating: int | None = None,
    opponent_rating_difference: int | None = 500,
    challenge_username: str | None = None,
    accept_casual_challenges: bool = False,
    selection_policy: str = "sample_probability_power",
    min_probability: float = 0.20,
    below_threshold_weight_scale: float = 0.25,
    probability_exponent: float = 2.0,
) -> dict[str, Any] | None:
    _auth_headers(config)
    config_path = render_lichess_bot_config(
        config,
        checkpoint_path,
        allow_matchmaking=allow_matchmaking,
        rated=rated,
        initial_time=initial_time,
        increment=increment,
        concurrency=concurrency,
        allow_during_games=allow_matchmaking,
        opponent_min_rating=opponent_min_rating,
        opponent_max_rating=opponent_max_rating,
        opponent_rating_difference=opponent_rating_difference,
        accept_bot=challenge_username is None,
        accept_casual_challenges=accept_casual_challenges,
        selection_policy=selection_policy,
        min_probability=min_probability,
        below_threshold_weight_scale=below_threshold_weight_scale,
        probability_exponent=probability_exponent,
    )
    root = project_root(config)
    command = [sys.executable, str(root / "third_party" / "lichess-bot" / "lichess-bot.py"), "--config", str(config_path)]

    if challenge_username and not max_games:
        raise ValueError("Direct challenge mode requires --max-games.")

    if not max_games and not max_runtime_seconds and not challenge_username:
        subprocess.run(command, check=True, cwd=root)
        return None

    logs_dir = root / "lichess_bot_auto_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    if challenge_username:
        log_name = "lichess-bot-direct-challenge.log"
    else:
        log_name = "lichess-bot-max-games.log" if max_games else "lichess-bot-timed-run.log"
    log_path = logs_dir / log_name
    log_handle = log_path.open("a", encoding="utf-8")

    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = subprocess.Popen(
        command,
        cwd=root,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )

    def _stop_process() -> None:
        try:
            if creationflags:
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
            process.wait(timeout=30)
        except Exception:
            process.kill()
            process.wait(timeout=30)

    if max_runtime_seconds:
        deadline = time.time() + max_runtime_seconds
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"lichess-bot exited early with code {process.returncode}. See {log_path}.")
                remaining_seconds = max(0, int(deadline - time.time()))
                print(f"Timed run active. Remaining seconds: {remaining_seconds}", flush=True)
                if remaining_seconds <= 0:
                    break
                time.sleep(min(60, remaining_seconds))
            _stop_process()
            return {
                "timed_run_seconds": max_runtime_seconds,
                "log_path": str(log_path),
            }
        finally:
            log_handle.close()

    if challenge_username:
        try:
            return _run_direct_challenge_series(
                process,
                config,
                username=challenge_username,
                rated=rated,
                initial_time=initial_time,
                increment=increment,
                max_games=max_games or 0,
                log_path=log_path,
            )
        finally:
            _stop_process()
            log_handle.close()

    initial_state = fetch_account_state(config)
    initial_account = initial_state["account"]
    initial_rapid = _rapid_perf(initial_account)
    username = initial_account["username"]
    baseline_games = int(initial_rapid.get("games", 0) or 0)
    target_games = baseline_games + max_games

    try:
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"lichess-bot exited early with code {process.returncode}. See {log_path}.")

            state = fetch_account_state(config)
            account = state["account"]
            rapid = _rapid_perf(account)
            current_games = int(rapid.get("games", 0) or 0)
            current_rating = rapid.get("rating")
            active_games = _active_game_count(state["playing"])
            progress = current_games - baseline_games
            print(
                f"User={username} rapid_games={current_games} progress={progress}/{max_games} "
                f"rapid_rating={current_rating} active_games={active_games}",
                flush=True,
            )

            if current_games >= target_games and active_games == 0:
                break
            time.sleep(poll_interval_seconds)

        _stop_process()

        final_state = fetch_account_state(config)
        final_account = final_state["account"]
        final_rapid = _rapid_perf(final_account)
        return {
            "username": final_account["username"],
            "games_started_from": baseline_games,
            "games_completed": int(final_rapid.get("games", 0) or 0) - baseline_games,
            "rapid_games_total": int(final_rapid.get("games", 0) or 0),
            "rapid_rating": final_rapid.get("rating"),
            "rapid_provisional": bool(final_rapid.get("prov")),
            "log_path": str(log_path),
            "profile_url": final_account.get("url"),
        }
    finally:
        log_handle.close()
