from __future__ import annotations

import argparse
from pathlib import Path
import sys

import chess

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from restible_bot.config import load_config
    from restible_bot.inference import load_inference_model, rank_moves
else:
    from .config import load_config
    from .inference import load_inference_model, rank_moves


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


def serve_uci(config_path: str | Path, checkpoint_path: str | Path) -> None:
    config = load_config(config_path)
    checkpoint = Path(checkpoint_path).resolve()
    model, _device = load_inference_model(checkpoint)
    target_self_elo = int(config["training"]["target_self_elo"])
    board = chess.Board()
    options = {
        "Threads": 1,
        "Move Overhead": 200,
    }

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        command = line.strip()
        if command == "uci":
            print("id name restible-maia2")
            print("id author Codex")
            print("option name Threads type spin default 1 min 1 max 128")
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
            board = _parse_position(command)
        elif command.startswith("go"):
            inference = rank_moves(model, board.fen(), target_self_elo, target_self_elo)
            best_move = inference["best_move"]
            best_probability = inference["moves"][0]["probability"]
            print(f"info depth 1 nodes 1 score cp 0 pv {best_move} string prob={best_probability:.4f}")
            print(f"bestmove {best_move}")
            sys.stdout.flush()
        elif command in {"stop", "ponderhit"}:
            continue
        elif command == "quit":
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/restible.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    serve_uci(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
