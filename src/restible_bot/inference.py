from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import random
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


def _post_move_win_probabilities(
    model: Maia2MoveModel,
    board: chess.Board,
    elo_self: int,
    elo_oppo: int,
) -> dict[str, float]:
    _ensure_maia2_path()
    from maia2.utils import board_to_tensor, map_to_category

    resources = maia2_resources()
    elo_dict = resources["elo_dict"]
    device = next(model.parameters()).device

    legal_moves = [move.uci() for move in board.legal_moves]
    if not legal_moves:
        return {}

    board_tensors: list[torch.Tensor] = []
    for move_uci in legal_moves:
        next_board = board.copy(stack=False)
        next_board.push_uci(move_uci)
        oriented_next_board = next_board if next_board.turn == chess.WHITE else next_board.mirror()
        board_tensors.append(board_to_tensor(oriented_next_board))

    boards_tensor = torch.stack(board_tensors, dim=0).to(device)
    next_self_bucket = map_to_category(elo_oppo, elo_dict)
    next_oppo_bucket = map_to_category(elo_self, elo_dict)
    elos_self = torch.full((len(legal_moves),), next_self_bucket, dtype=torch.long, device=device)
    elos_oppo = torch.full((len(legal_moves),), next_oppo_bucket, dtype=torch.long, device=device)

    with torch.no_grad():
        _, _, logits_value = model.backbone(boards_tensor, elos_self, elos_oppo)
        opponent_scores = (logits_value / 2 + 0.5).clamp(0, 1)
        mover_scores = 1.0 - opponent_scores

    return {move_uci: float(score.item()) for move_uci, score in zip(legal_moves, mover_scores)}


def _terminal_score_for_color(board: chess.Board, color: chess.Color) -> float | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == color else 0.0


@lru_cache(maxsize=300_000)
def _leaf_active_win_probability(
    model: Maia2MoveModel,
    fen: str,
    elo_self: int,
    elo_oppo: int,
) -> float:
    _ensure_maia2_path()
    from maia2.utils import board_to_tensor, map_to_category

    resources = maia2_resources()
    elo_dict = resources["elo_dict"]
    board = chess.Board(fen)
    oriented_board = board if board.turn == chess.WHITE else board.mirror()
    board_tensor = board_to_tensor(oriented_board).unsqueeze(0).to(next(model.parameters()).device)
    elo_self_bucket = torch.tensor(
        [map_to_category(elo_self, elo_dict)],
        dtype=torch.long,
        device=board_tensor.device,
    )
    elo_oppo_bucket = torch.tensor(
        [map_to_category(elo_oppo, elo_dict)],
        dtype=torch.long,
        device=board_tensor.device,
    )
    with torch.no_grad():
        _policy_a, _policy_b, logits_value = model.backbone(board_tensor, elo_self_bucket, elo_oppo_bucket)
    return float((logits_value / 2 + 0.5).clamp(0, 1).item())


def _leaf_win_for_color(
    model: Maia2MoveModel,
    board: chess.Board,
    color: chess.Color,
    elo_self: int,
    elo_oppo: int,
) -> float:
    terminal = _terminal_score_for_color(board, color)
    if terminal is not None:
        return terminal
    active_win = _leaf_active_win_probability(model, board.fen(), elo_self, elo_oppo)
    return active_win if board.turn == color else 1.0 - active_win


def _rank_policy_moves(
    model: Maia2MoveModel,
    board: chess.Board,
    elo_self: int,
    elo_oppo: int,
) -> list[dict[str, Any]]:
    _ensure_maia2_path()
    from maia2.utils import board_to_tensor, map_to_category, mirror_move

    resources = maia2_resources()
    all_moves_dict = resources["all_moves_dict"]
    all_moves_dict_reversed = resources["all_moves_dict_reversed"]
    elo_dict = resources["elo_dict"]

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
    return ranked_moves


@lru_cache(maxsize=300_000)
def _top_policy_moves(
    model: Maia2MoveModel,
    fen: str,
    elo_self: int,
    elo_oppo: int,
    top_k: int,
) -> tuple[tuple[str, float], ...]:
    board = chess.Board(fen)
    ranked_moves = _rank_policy_moves(model, board, elo_self, elo_oppo)
    return tuple((move["uci"], float(move["probability"])) for move in ranked_moves[:top_k])


@lru_cache(maxsize=300_000)
def _explored_win_for_color(
    model: Maia2MoveModel,
    fen: str,
    color_is_white: bool,
    depth: int,
    elo_self: int,
    elo_oppo: int,
    top_k: int,
    probability_exponent: float,
) -> float:
    board = chess.Board(fen)
    color = chess.WHITE if color_is_white else chess.BLACK
    terminal = _terminal_score_for_color(board, color)
    if terminal is not None:
        return terminal
    if depth <= 0:
        return _leaf_win_for_color(model, board, color, elo_self, elo_oppo)

    moves = _top_policy_moves(model, fen, elo_self, elo_oppo, top_k)
    if not moves:
        return _leaf_win_for_color(model, board, color, elo_self, elo_oppo)

    weights = [max(probability, 0.0) ** probability_exponent for _uci, probability in moves]
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0 for _move in moves]
        total_weight = float(len(moves))

    expected_win = 0.0
    for (move_uci, _probability), weight in zip(moves, weights):
        child = board.copy(stack=False)
        child.push_uci(move_uci)
        expected_win += (weight / total_weight) * _explored_win_for_color(
            model,
            child.fen(),
            color_is_white,
            depth - 1,
            elo_self,
            elo_oppo,
            top_k,
            probability_exponent,
        )
    return expected_win


def _add_3ply_win_probability_weights(
    model: Maia2MoveModel,
    board: chess.Board,
    ranked_moves: list[dict[str, Any]],
    elo_self: int,
    elo_oppo: int,
    *,
    search_top_k: int,
    search_plies: int,
    probability_exponent: float,
) -> list[dict[str, Any]]:
    sample_pool = ranked_moves[: min(search_top_k, len(ranked_moves))]
    root_color_is_white = board.turn == chess.WHITE
    depth_after_root = max(0, search_plies - 1)
    for move in sample_pool:
        child = board.copy(stack=False)
        child.push_uci(move["uci"])
        explored_win_probability = _explored_win_for_color(
            model,
            child.fen(),
            root_color_is_white,
            depth_after_root,
            elo_self,
            elo_oppo,
            search_top_k,
            probability_exponent,
        )
        move["explored_win_probability"] = explored_win_probability
        move["selection_weight"] = (max(move["probability"], 0.0) ** probability_exponent) * explored_win_probability
    return sample_pool


def _select_move(
    ranked_moves: list[dict[str, Any]],
    *,
    selection_policy: str,
    min_probability: float,
    below_threshold_weight_scale: float,
    probability_exponent: float,
) -> tuple[str, list[dict[str, Any]]]:
    if not ranked_moves:
        raise ValueError("ranked_moves must not be empty")

    if selection_policy == "top1":
        return ranked_moves[0]["uci"], ranked_moves[:1]
    if selection_policy == "sample_top2":
        sample_pool = ranked_moves[: min(2, len(ranked_moves))]
        weights = [move["probability"] for move in sample_pool]
        if sum(weights) <= 0:
            weights = None
        selected_move = random.choices(sample_pool, weights=weights, k=1)[0]["uci"]
        return selected_move, sample_pool

    if selection_policy == "sample_top3":
        sample_pool = ranked_moves[: min(3, len(ranked_moves))]
        weights = [move["probability"] for move in sample_pool]
    elif selection_policy == "sample_min_probability":
        sample_pool = [move for move in ranked_moves if move["probability"] >= min_probability]
        if not sample_pool:
            sample_pool = ranked_moves[:1]
        weights = [move["probability"] for move in sample_pool]
    elif selection_policy == "sample_reweighted_below_threshold":
        sample_pool = list(ranked_moves)
        weights = [
            move["probability"] if move["probability"] >= min_probability else move["probability"] * below_threshold_weight_scale
            for move in sample_pool
        ]
    elif selection_policy == "sample_probability_power":
        sample_pool = list(ranked_moves)
        weights = [move["probability"] ** probability_exponent for move in sample_pool]
    elif selection_policy == "sample_probability_times_win_probability":
        sample_pool = list(ranked_moves)
        weights = [move["selection_weight"] for move in sample_pool]
    elif selection_policy == "sample_probability_power_3ply_win_probability":
        sample_pool = [move for move in ranked_moves if "selection_weight" in move]
        weights = [move["selection_weight"] for move in sample_pool]
    else:
        raise ValueError(f"Unsupported selection_policy: {selection_policy}")

    if sum(weights) <= 0:
        weights = None
    selected_move = random.choices(sample_pool, weights=weights, k=1)[0]["uci"]
    return selected_move, sample_pool


def rank_moves(
    model: Maia2MoveModel,
    fen: str,
    elo_self: int,
    elo_oppo: int,
    *,
    selection_policy: str = "sample_probability_power",
    min_probability: float = 0.20,
    below_threshold_weight_scale: float = 0.25,
    probability_exponent: float = 2.0,
    search_top_k: int = 3,
    search_plies: int = 3,
) -> dict[str, Any]:
    board = chess.Board(fen)
    ranked_moves = _rank_policy_moves(model, board, elo_self, elo_oppo)
    if selection_policy == "sample_probability_times_win_probability":
        move_win_probabilities = _post_move_win_probabilities(model, board, elo_self, elo_oppo)
        for move in ranked_moves:
            move["post_move_win_probability"] = move_win_probabilities[move["uci"]]
            move["selection_weight"] = move["probability"] * move["post_move_win_probability"]
    elif selection_policy == "sample_probability_power_3ply_win_probability":
        _add_3ply_win_probability_weights(
            model,
            board,
            ranked_moves,
            elo_self,
            elo_oppo,
            search_top_k=search_top_k,
            search_plies=search_plies,
            probability_exponent=probability_exponent,
        )
    selected_move, sample_pool = _select_move(
        ranked_moves,
        selection_policy=selection_policy,
        min_probability=min_probability,
        below_threshold_weight_scale=below_threshold_weight_scale,
        probability_exponent=probability_exponent,
    )
    return {
        "fen": fen,
        "best_move": selected_move,
        "top_move": ranked_moves[0]["uci"],
        "sample_pool": sample_pool,
        "selection_policy": selection_policy,
        "min_probability": min_probability,
        "below_threshold_weight_scale": below_threshold_weight_scale,
        "probability_exponent": probability_exponent,
        "search_top_k": search_top_k,
        "search_plies": search_plies,
        "moves": ranked_moves,
    }
