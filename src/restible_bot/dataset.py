from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import random
import sys
from typing import Any

import chess
import chess.pgn
import torch

from .config import data_dir
from .lichess_export import latest_raw_export
from .utils import read_jsonl, write_json, write_jsonl


def _ensure_maia2_path() -> None:
    maia2_root = Path(__file__).resolve().parents[2] / "third_party" / "maia2"
    if str(maia2_root) not in sys.path:
        sys.path.insert(0, str(maia2_root))


def _prepared_dir(config: dict[str, Any]) -> Path:
    return data_dir(config) / "prepared"


def _splits_dir(config: dict[str, Any]) -> Path:
    return data_dir(config) / "splits"


def _iter_games(pgn_path: Path):
    with pgn_path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            yield game


def _is_rapid_game(game: chess.pgn.Game) -> bool:
    speed = (game.headers.get("Speed") or "").lower()
    event = (game.headers.get("Event") or "").lower()
    return speed == "rapid" or "rapid" in event


def _parse_timestamp(headers: chess.pgn.Headers) -> tuple[str, str, str]:
    utc_date = headers.get("UTCDate")
    utc_time = headers.get("UTCTime")
    if not utc_date or not utc_time:
        raise ValueError("Missing UTCDate or UTCTime in PGN headers.")
    return f"{utc_date} {utc_time}", utc_date, utc_time


def _parse_player_context(game: chess.pgn.Game, username: str, minimum_elo: int) -> tuple[bool, int, int, str]:
    normalized = username.lower()
    white_name = (game.headers.get("White") or "").lower()
    black_name = (game.headers.get("Black") or "").lower()
    if normalized == white_name:
        player_elo_raw = game.headers.get("WhiteElo")
        opponent_elo_raw = game.headers.get("BlackElo")
        return chess.WHITE, _parse_elo(player_elo_raw, minimum_elo), _parse_optional_elo(opponent_elo_raw), "white"
    if normalized == black_name:
        player_elo_raw = game.headers.get("BlackElo")
        opponent_elo_raw = game.headers.get("WhiteElo")
        return chess.BLACK, _parse_elo(player_elo_raw, minimum_elo), _parse_optional_elo(opponent_elo_raw), "black"
    raise ValueError("Configured player not found in PGN headers.")


def _parse_elo(value: str | None, minimum_elo: int) -> int:
    if not value:
        raise ValueError("Missing player Elo.")
    elo = int(value)
    if elo < minimum_elo:
        raise ValueError("Player Elo below threshold.")
    return elo


def _parse_optional_elo(value: str | None) -> int:
    try:
        return int(value) if value else 1900
    except ValueError:
        return 1900


def _game_id(headers: chess.pgn.Headers) -> str:
    site = headers.get("Site") or ""
    if site:
        return site.rstrip("/").split("/")[-1]
    return f"game-{headers.get('UTCDate', 'unknown')}-{headers.get('UTCTime', 'unknown')}"


def _build_game_record(game: chess.pgn.Game, username: str, minimum_elo: int, source_pgn: Path) -> dict[str, Any]:
    sort_key, utc_date, utc_time = _parse_timestamp(game.headers)
    color, player_elo, opponent_elo, color_name = _parse_player_context(game, username, minimum_elo)

    board = game.board()
    positions: list[dict[str, Any]] = []
    for ply, move in enumerate(game.mainline_moves()):
        if board.turn == color:
            positions.append(
                {
                    "fen": board.fen(),
                    "move_uci": move.uci(),
                    "move_index": ply,
                }
            )
        board.push(move)

    if not positions:
        raise ValueError("Game has no player-to-move positions after filtering.")

    return {
        "game_id": _game_id(game.headers),
        "site": game.headers.get("Site"),
        "color": color_name,
        "restible_elo": player_elo,
        "opponent_elo": opponent_elo,
        "utc_date": utc_date,
        "utc_time": utc_time,
        "sort_key": sort_key,
        "source_pgn": str(source_pgn),
        "positions": positions,
    }


def _split_games(
    games: list[dict[str, Any]],
    test_fraction: float,
    validation_fraction_within_train: float,
    split_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not games:
        raise RuntimeError("No eligible games were available to split.")

    shuffled_games = list(games)
    random.Random(split_seed).shuffle(shuffled_games)

    total_games = len(games)
    test_count = max(1, math.ceil(total_games * test_fraction))
    if total_games > 1:
        test_count = min(test_count, total_games - 1)
    test_start = total_games - test_count
    development_games = shuffled_games[:test_start]
    test_games = shuffled_games[test_start:]

    if not development_games:
        return [], [], test_games

    val_count = 0
    if len(development_games) > 1:
        val_count = max(1, math.ceil(len(development_games) * validation_fraction_within_train))
        val_count = min(val_count, len(development_games) - 1)

    if val_count:
        train_games = development_games[:-val_count]
        val_games = development_games[-val_count:]
    else:
        train_games = development_games
        val_games = []
    return train_games, val_games, test_games


def prepare_dataset(config: dict[str, Any], raw_pgn: Path | None = None) -> dict[str, Path]:
    raw_pgn = raw_pgn or latest_raw_export(config)
    username = config["lichess"]["username"]
    minimum_elo = int(config["dataset"]["min_player_elo"])

    eligible_games: list[dict[str, Any]] = []
    skipped = {
        "non_rapid": 0,
        "missing_player": 0,
        "missing_or_low_elo": 0,
        "missing_time": 0,
        "empty_positions": 0,
    }

    for game in _iter_games(raw_pgn):
        if not _is_rapid_game(game):
            skipped["non_rapid"] += 1
            continue
        try:
            record = _build_game_record(game, username, minimum_elo, raw_pgn)
        except ValueError as exc:
            message = str(exc)
            if "UTCDate" in message or "UTCTime" in message:
                skipped["missing_time"] += 1
            elif "Configured player" in message:
                skipped["missing_player"] += 1
            elif "player Elo" in message or "below threshold" in message:
                skipped["missing_or_low_elo"] += 1
            else:
                skipped["empty_positions"] += 1
            continue
        eligible_games.append(record)

    train_games, val_games, test_games = _split_games(
        eligible_games,
        float(config["dataset"]["test_fraction"]),
        float(config["dataset"]["validation_fraction_within_train"]),
        int(config["dataset"]["split_seed"]),
    )

    prepared_path = _prepared_dir(config) / "restible_games.jsonl"
    train_path = _splits_dir(config) / "train_games.jsonl"
    val_path = _splits_dir(config) / "val_games.jsonl"
    test_path = _splits_dir(config) / "test_games.jsonl"
    summary_path = _splits_dir(config) / "split_summary.json"

    write_jsonl(prepared_path, eligible_games)
    write_jsonl(train_path, train_games)
    write_jsonl(val_path, val_games)
    write_jsonl(test_path, test_games)

    def _position_count(records: list[dict[str, Any]]) -> int:
        return sum(len(record["positions"]) for record in records)

    summary = {
        "raw_pgn": str(raw_pgn),
        "eligible_games": len(eligible_games),
        "eligible_positions": _position_count(eligible_games),
        "train_games": len(train_games),
        "train_positions": _position_count(train_games),
        "val_games": len(val_games),
        "val_positions": _position_count(val_games),
        "test_games": len(test_games),
        "test_positions": _position_count(test_games),
        "test_fraction_requested": float(config["dataset"]["test_fraction"]),
        "validation_fraction_within_train_requested": float(config["dataset"]["validation_fraction_within_train"]),
        "split_strategy": "random_by_game",
        "split_seed": int(config["dataset"]["split_seed"]),
        "skipped": skipped,
    }
    write_json(summary_path, summary)
    return {
        "prepared": prepared_path,
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "summary": summary_path,
    }


def latest_split_paths(config: dict[str, Any]) -> dict[str, Path]:
    base = _splits_dir(config)
    paths = {
        "train": base / "train_games.jsonl",
        "val": base / "val_games.jsonl",
        "test": base / "test_games.jsonl",
        "summary": base / "split_summary.json",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Missing split artifacts: {missing_str}. Run prepare-dataset first.")
    return paths


@lru_cache(maxsize=1)
def maia2_resources() -> dict[str, Any]:
    _ensure_maia2_path()
    from maia2.utils import create_elo_dict, get_all_possible_moves

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: index for index, move in enumerate(all_moves)}
    all_moves_dict_reversed = {index: move for move, index in all_moves_dict.items()}
    elo_dict = create_elo_dict()
    return {
        "all_moves_dict": all_moves_dict,
        "all_moves_dict_reversed": all_moves_dict_reversed,
        "elo_dict": elo_dict,
    }


def _orient_position(fen: str, move_uci: str) -> tuple[chess.Board, str]:
    _ensure_maia2_path()
    from maia2.utils import mirror_move

    board = chess.Board(fen)
    if board.turn == chess.BLACK:
        return board.mirror(), mirror_move(move_uci)
    return board, move_uci


def _position_sample(
    fen: str,
    move_uci: str,
    restible_elo: int,
    opponent_elo: int,
) -> tuple[torch.Tensor, int, int, int, torch.Tensor]:
    _ensure_maia2_path()
    from maia2.utils import board_to_tensor, map_to_category

    resources = maia2_resources()
    board, oriented_move = _orient_position(fen, move_uci)
    all_moves_dict = resources["all_moves_dict"]
    elo_dict = resources["elo_dict"]

    board_tensor = board_to_tensor(board)
    label = all_moves_dict[oriented_move]
    elo_self_bucket = map_to_category(restible_elo, elo_dict)
    elo_oppo_bucket = map_to_category(opponent_elo, elo_dict)
    legal_moves = torch.zeros(len(all_moves_dict), dtype=torch.float32)
    legal_indices = [all_moves_dict[legal_move.uci()] for legal_move in board.legal_moves]
    legal_moves[legal_indices] = 1.0
    return board_tensor, label, elo_self_bucket, elo_oppo_bucket, legal_moves


class RestibleMoveDataset(torch.utils.data.Dataset):
    def __init__(self, split_path: Path, limit_positions: int = 0) -> None:
        records = read_jsonl(split_path)
        samples: list[dict[str, Any]] = []
        for record in records:
            for position in record["positions"]:
                samples.append(
                    {
                        "game_id": record["game_id"],
                        "fen": position["fen"],
                        "move_uci": position["move_uci"],
                        "move_index": int(position["move_index"]),
                        "restible_elo": int(record["restible_elo"]),
                        "opponent_elo": int(record["opponent_elo"]),
                    }
                )
                if limit_positions and len(samples) >= limit_positions:
                    break
            if limit_positions and len(samples) >= limit_positions:
                break
        if not samples:
            raise RuntimeError(f"No training samples could be built from {split_path}.")
        self.split_path = split_path
        self.samples = samples
        self.game_count = len({sample["game_id"] for sample in samples})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        board_tensor, label, elo_self_bucket, elo_oppo_bucket, legal_moves = _position_sample(
            sample["fen"],
            sample["move_uci"],
            sample["restible_elo"],
            sample["opponent_elo"],
        )
        return (
            board_tensor,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(elo_self_bucket, dtype=torch.long),
            torch.tensor(elo_oppo_bucket, dtype=torch.long),
            legal_moves,
            sample["game_id"],
            torch.tensor(sample["move_index"], dtype=torch.long),
        )
