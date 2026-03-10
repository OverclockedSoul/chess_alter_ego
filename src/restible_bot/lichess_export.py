from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess.pgn
import requests

from .config import data_dir
from .utils import utc_timestamp, write_json


def _raw_data_dir(config: dict[str, Any]) -> Path:
    return data_dir(config) / "raw"


def _is_rapid_game(game: chess.pgn.Game) -> bool:
    speed = (game.headers.get("Speed") or "").lower()
    event = (game.headers.get("Event") or "").lower()
    return speed == "rapid" or "rapid" in event


def _game_timestamp_ms(game: chess.pgn.Game) -> int | None:
    utc_date = game.headers.get("UTCDate")
    utc_time = game.headers.get("UTCTime")
    if not utc_date or not utc_time:
        return None
    try:
        parsed = datetime.strptime(f"{utc_date} {utc_time}", "%Y.%m.%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(parsed.timestamp() * 1000)


def _iter_pgn_games(payload: str):
    handle = io.StringIO(payload)
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            break
        yield game


def export_games(config: dict[str, Any]) -> dict[str, Path]:
    settings = config["lichess"]
    username = settings["username"]
    destination = _raw_data_dir(config) / f"{username}_all_rapid_{utc_timestamp()}.pgn"
    metadata_path = destination.with_suffix(".metadata.json")

    token = os.getenv(settings["token_env"])
    headers = {"Accept": "application/x-chess-pgn"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{settings['base_url'].rstrip('/')}/api/games/user/{username}"
    batch_size = int(settings.get("request_batch_size", 300))
    total_games = 0
    rapid_games = 0
    batches = 0
    next_until: int | None = None

    with destination.open("w", encoding="utf-8") as output_handle:
        session = requests.Session()
        while True:
            params: dict[str, Any] = {
                "max": batch_size,
                "moves": "true",
                "pgnInJson": "false",
                "opening": "true",
            }
            if next_until is not None:
                params["until"] = next_until

            response = session.get(url, headers=headers, params=params, timeout=120)
            if response.status_code in {401, 403} and not token:
                raise RuntimeError(
                    f"Export requires authentication. Set {settings['token_env']} in .env and retry."
                )
            response.raise_for_status()
            payload = response.text.strip()
            if not payload:
                break

            oldest_timestamp: int | None = None
            batch_games = 0
            batches += 1
            for game in _iter_pgn_games(payload):
                batch_games += 1
                total_games += 1
                timestamp = _game_timestamp_ms(game)
                if timestamp is not None and (oldest_timestamp is None or timestamp < oldest_timestamp):
                    oldest_timestamp = timestamp
                if not _is_rapid_game(game):
                    continue
                rapid_games += 1
                print(game, file=output_handle, end="\n\n")

            if batch_games < batch_size or oldest_timestamp is None:
                break
            next_until = oldest_timestamp - 1

    if rapid_games == 0:
        raise RuntimeError("Lichess export completed but no rapid games were written.")

    write_json(
        metadata_path,
        {
            "username": username,
            "source_url": url,
            "fetched_at": utc_timestamp(),
            "batches": batches,
            "total_games_seen": total_games,
            "rapid_games_written": rapid_games,
            "used_authentication": bool(token),
            "batch_size": batch_size,
            "filter_mode": "local_pgn_header_filter",
        },
    )
    return {"pgn": destination, "metadata": metadata_path}


def latest_raw_export(config: dict[str, Any]) -> Path:
    username = config["lichess"]["username"]
    exports = sorted(_raw_data_dir(config).glob(f"{username}_all_rapid_*.pgn"))
    if not exports:
        raise FileNotFoundError("No rapid PGN export found. Run export-games first.")
    return exports[-1]

