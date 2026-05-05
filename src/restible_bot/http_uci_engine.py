from __future__ import annotations

import os
import random
import sys
import time
from typing import Any

import chess
import requests


DEFAULT_CHESS_BOT_URL = "http://127.0.0.1:8001"


def _parse_position(command: str) -> chess.Board:
    tokens = command.strip().split()
    if len(tokens) < 2:
        return chess.Board()

    if tokens[1] == "startpos":
        board = chess.Board()
        move_start = 3 if len(tokens) > 2 and tokens[2] == "moves" else len(tokens)
    elif tokens[1] == "fen":
        moves_index = tokens.index("moves") if "moves" in tokens else len(tokens)
        fen = " ".join(tokens[2:moves_index])
        board = chess.Board(fen)
        move_start = moves_index + 1 if moves_index < len(tokens) else len(tokens)
    else:
        return chess.Board()

    for move_uci in tokens[move_start:]:
        try:
            board.push_uci(move_uci)
        except ValueError:
            break
    return board


def _probability(payload: dict[str, Any], name: str) -> float | None:
    value = payload.get(name)
    if not isinstance(value, int | float) or not 0.0 <= float(value) <= 1.0:
        return None
    return float(value)


def _request_move(board: chess.Board) -> tuple[str, float | None, float | None, int | None]:
    base_url = os.getenv("CHESS_BOT_URL", DEFAULT_CHESS_BOT_URL).rstrip("/")
    timeout = float(os.getenv("CHESS_BOT_HTTP_TIMEOUT", "30"))
    response = requests.post(f"{base_url}/move", json={"fen": board.fen()}, timeout=timeout)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    move = payload.get("move")
    if not isinstance(move, str) or not move:
        raise RuntimeError("Chess engine API response did not contain a move.")
    legal_move_count = payload.get("legalMoveCount")
    if not isinstance(legal_move_count, int) or legal_move_count < 1:
        legal_move_count = None
    return move, _probability(payload, "topProbability"), _probability(payload, "secondProbability"), legal_move_count


def _move_delay_seconds(
    board: chess.Board,
    top_probability: float | None,
    second_probability: float | None,
    legal_move_count: int | None,
) -> float:
    if top_probability is None:
        return 0.0
    cap = 2.0 if board.fullmove_number <= 10 else 5.0
    if legal_move_count == 1:
        return min(0.5, cap)
    if second_probability is None:
        return min((1.0 - top_probability) * 5.0, cap)

    uncertainty = 1.0 - min(max((top_probability - second_probability) / 0.35, 0.0), 1.0)
    delay = 1.0 + uncertainty * 4.0 + random.uniform(-0.4, 0.8)
    return max(0.6, min(delay, cap))


def serve_uci() -> None:
    board = chess.Board()
    options = {
        "Move Overhead": 200,
    }

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        command = line.strip()
        if command == "uci":
            print("id name restible-http-uci")
            print("id author Codex")
            print("option name Move Overhead type spin default 200 min 0 max 5000")
            print("uciok")
            sys.stdout.flush()
        elif command == "isready":
            print("readyok")
            sys.stdout.flush()
        elif command.startswith("setoption"):
            parts = command.split()
            if "name" in parts:
                name_index = parts.index("name") + 1
                value_index = parts.index("value") if "value" in parts else len(parts)
                option_name = " ".join(parts[name_index:value_index])
                option_value = " ".join(parts[value_index + 1 :]) if value_index < len(parts) else ""
                if option_name in options and option_value:
                    try:
                        options[option_name] = int(option_value)
                    except ValueError:
                        pass
        elif command == "ucinewgame":
            board = chess.Board()
        elif command.startswith("position"):
            try:
                board = _parse_position(command)
            except ValueError as error:
                print(f"info string invalid position: {error}")
                sys.stdout.flush()
        elif command.startswith("go"):
            try:
                best_move, top_probability, second_probability, legal_move_count = _request_move(board)
                chess.Move.from_uci(best_move)
                wait_seconds = _move_delay_seconds(board, top_probability, second_probability, legal_move_count)
                if top_probability is not None:
                    second_text = "none" if second_probability is None else f"{second_probability:.4f}"
                    print(
                        f"info string top_probability={top_probability:.4f} "
                        f"second_probability={second_text} move_delay={wait_seconds:.2f}s"
                    )
                    sys.stdout.flush()
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
            except Exception as error:
                print(f"info string http engine error: {error}")
                best_move = "0000"
            print(f"bestmove {best_move}")
            sys.stdout.flush()
        elif command in {"stop", "ponderhit"}:
            continue
        elif command == "quit":
            break


def main() -> None:
    serve_uci()


if __name__ == "__main__":
    main()
