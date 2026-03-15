from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import data_dir
from .dataset import RestibleMoveDataset, latest_split_paths
from .maia2_model import Maia2MoveModel, load_checkpoint_model
from .utils import detect_device, write_json


def _reports_dir(config: dict[str, Any]) -> Path:
    return data_dir(config) / "reports"


def _output_suffix(config: dict[str, Any]) -> str:
    return str(config["training"].get("output_suffix") or "").strip()


def _report_file_name(config: dict[str, Any], base_name: str) -> str:
    suffix = _output_suffix(config)
    if not suffix:
        return base_name
    stem, extension = base_name.rsplit(".", 1)
    return f"{stem}_{suffix}.{extension}"


def _phase_name(ply: int) -> str:
    if ply <= 15:
        return "opening"
    if ply <= 39:
        return "middlegame"
    return "late_game"


def evaluate_model(
    model: Maia2MoveModel,
    dataset: RestibleMoveDataset,
    device: torch.device,
    split: str,
    batch_size: int,
) -> dict[str, Any]:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    totals = {
        "top1": {"correct": 0, "total": 0},
        "top3": {"correct": 0, "total": 0},
        "top5": {"correct": 0, "total": 0},
    }
    phase_metrics = {
        "opening": {"positions": 0, "top1": 0, "top3": 0, "top5": 0},
        "middlegame": {"positions": 0, "top1": 0, "top3": 0, "top5": 0},
        "late_game": {"positions": 0, "top1": 0, "top3": 0, "top5": 0},
    }
    total_true_probability = 0.0

    model.eval()
    with torch.no_grad():
        for boards, labels, elos_self, elos_oppo, legal_moves, _game_ids, move_indices in dataloader:
            boards = boards.to(device)
            labels = labels.to(device)
            elos_self = elos_self.to(device)
            elos_oppo = elos_oppo.to(device)
            legal_moves = legal_moves.to(device)

            logits = model(boards, elos_self, elos_oppo)
            masked_logits = logits.masked_fill(legal_moves <= 0, torch.finfo(logits.dtype).min)
            probabilities = torch.softmax(masked_logits, dim=-1)
            total_true_probability += probabilities.gather(1, labels.unsqueeze(1)).sum().item()

            top1 = torch.topk(masked_logits, k=1, dim=-1).indices
            top3 = torch.topk(masked_logits, k=3, dim=-1).indices
            top5 = torch.topk(masked_logits, k=5, dim=-1).indices
            top1_hits = (top1.squeeze(1) == labels)
            top3_hits = (top3 == labels.unsqueeze(1)).any(dim=1)
            top5_hits = (top5 == labels.unsqueeze(1)).any(dim=1)

            batch_size_actual = labels.size(0)
            totals["top1"]["correct"] += int(top1_hits.sum().item())
            totals["top3"]["correct"] += int(top3_hits.sum().item())
            totals["top5"]["correct"] += int(top5_hits.sum().item())
            totals["top1"]["total"] += batch_size_actual
            totals["top3"]["total"] += batch_size_actual
            totals["top5"]["total"] += batch_size_actual

            for index in range(batch_size_actual):
                phase = _phase_name(int(move_indices[index].item()))
                phase_metrics[phase]["positions"] += 1
                phase_metrics[phase]["top1"] += int(top1_hits[index].item())
                phase_metrics[phase]["top3"] += int(top3_hits[index].item())
                phase_metrics[phase]["top5"] += int(top5_hits[index].item())

    total_positions = totals["top1"]["total"]
    metrics = {
        "split": split,
        "games": dataset.game_count,
        "positions": total_positions,
        "top1": {
            "correct": totals["top1"]["correct"],
            "total": total_positions,
            "accuracy": round(totals["top1"]["correct"] / total_positions, 4),
        },
        "top3": {
            "correct": totals["top3"]["correct"],
            "total": total_positions,
            "accuracy": round(totals["top3"]["correct"] / total_positions, 4),
        },
        "top5": {
            "correct": totals["top5"]["correct"],
            "total": total_positions,
            "accuracy": round(totals["top5"]["correct"] / total_positions, 4),
        },
        "mean_true_move_probability": round(total_true_probability / total_positions, 4),
        "by_phase": {},
    }
    for phase, values in phase_metrics.items():
        positions = values["positions"]
        if positions == 0:
            continue
        metrics["by_phase"][phase] = {
            "positions": positions,
            "top1": round(values["top1"] / positions, 4),
            "top3": round(values["top3"] / positions, 4),
            "top5": round(values["top5"] / positions, 4),
        }
    return metrics


def _markdown_report(metrics: dict[str, Any]) -> str:
    lines = [
        f"# {metrics['split'].title()} Metrics",
        "",
        f"- Games: {metrics['games']}",
        f"- Positions: {metrics['positions']}",
        f"- Top-1 accuracy: {metrics['top1']['accuracy']:.4f} ({metrics['top1']['correct']}/{metrics['top1']['total']})",
        f"- Top-3 accuracy: {metrics['top3']['accuracy']:.4f} ({metrics['top3']['correct']}/{metrics['top3']['total']})",
        f"- Top-5 accuracy: {metrics['top5']['accuracy']:.4f} ({metrics['top5']['correct']}/{metrics['top5']['total']})",
        f"- Mean true move probability: {metrics['mean_true_move_probability']:.4f}",
    ]
    if metrics.get("by_phase"):
        lines.append("")
        lines.append("## By Phase")
        for phase, values in metrics["by_phase"].items():
            lines.append(
                f"- {phase}: positions={values['positions']}, top1={values['top1']:.4f}, "
                f"top3={values['top3']:.4f}, top5={values['top5']:.4f}"
            )
    return "\n".join(lines) + "\n"


def evaluate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: Path,
    split: str,
    device: torch.device | None = None,
) -> dict[str, Any]:
    paths = latest_split_paths(config)
    if split not in {"val", "test"}:
        raise ValueError("Supported splits are 'val' and 'test'.")

    device = device or detect_device()
    dataset = RestibleMoveDataset(
        paths[split],
        override_self_elo=config["training"].get("override_self_elo"),
        override_opponent_elo=config["training"].get("override_opponent_elo"),
    )
    model, _ = load_checkpoint_model(checkpoint_path=checkpoint_path, device=device)
    batch_size = 128 if device.type == "cuda" else 32
    metrics = evaluate_model(model, dataset, device=device, split=split, batch_size=batch_size)

    reports_dir = _reports_dir(config)
    json_path = reports_dir / _report_file_name(
        config,
        "validation_metrics.json" if split == "val" else "test_metrics.json",
    )
    write_json(json_path, metrics)
    if split == "test":
        markdown_path = reports_dir / _report_file_name(config, "test_metrics.md")
        markdown_path.write_text(_markdown_report(metrics), encoding="utf-8")
    return metrics
