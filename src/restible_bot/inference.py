from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import chess
import torch

from .dataset import maia2_resources
from .maia2_model import Maia2MoveModel, load_checkpoint_model
from .utils import detect_device


def _ensure_maia2_path() -> None:
    maia2_root = Path(__file__).resolve().parents[2] / "third_party" / "maia2"
    if str(maia2_root) not in sys.path:
        sys.path.insert(0, str(maia2_root))


def load_inference_model(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> tuple[Maia2MoveModel, torch.device]:
    device = device or detect_device()
    model, _ = load_checkpoint_model(checkpoint_path=checkpoint_path, device=device)
    model.eval()
    return model, device


def rank_moves(
    model: Maia2MoveModel,
    fen: str,
    elo_self: int,
    elo_oppo: int,
) -> dict[str, Any]:
    _ensure_maia2_path()
    from maia2.utils import board_to_tensor, map_to_category, mirror_move

    resources = maia2_resources()
    all_moves_dict = resources["all_moves_dict"]
    all_moves_dict_reversed = resources["all_moves_dict_reversed"]
    elo_dict = resources["elo_dict"]

    board = chess.Board(fen)
    oriented_board = board if board.turn == chess.WHITE else board.mirror()
    board_tensor = board_to_tensor(oriented_board).unsqueeze(0).to(next(model.parameters()).device)
    elo_self_bucket = torch.tensor([map_to_category(elo_self, elo_dict)], dtype=torch.long, device=board_tensor.device)
    elo_oppo_bucket = torch.tensor([map_to_category(elo_oppo, elo_dict)], dtype=torch.long, device=board_tensor.device)

    legal_moves_tensor = torch.zeros(len(all_moves_dict), dtype=torch.float32, device=board_tensor.device)
    legal_indices = [all_moves_dict[move.uci()] for move in oriented_board.legal_moves]
    legal_moves_tensor[legal_indices] = 1.0

    with torch.no_grad():
        logits = model(board_tensor, elo_self_bucket, elo_oppo_bucket).squeeze(0)
        masked_logits = logits.masked_fill(legal_moves_tensor <= 0, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(masked_logits, dim=-1)

    ranked_moves: list[dict[str, Any]] = []
    is_black = board.turn == chess.BLACK
    for move_index in legal_indices:
        oriented_move = all_moves_dict_reversed[move_index]
        move = mirror_move(oriented_move) if is_black else oriented_move
        ranked_moves.append(
            {
                "uci": move,
                "probability": float(probabilities[move_index].item()),
            }
        )
    ranked_moves.sort(key=lambda item: item["probability"], reverse=True)
    best_move = ranked_moves[0]["uci"]
    return {
        "fen": fen,
        "best_move": best_move,
        "moves": ranked_moves,
    }

