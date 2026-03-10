from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import data_dir
from .dataset import RestibleMoveDataset, latest_split_paths
from .evaluate import evaluate_checkpoint, evaluate_model
from .maia2_model import Maia2MoveModel, checkpoint_payload, load_pretrained_model
from .utils import detect_device, device_summary, write_json


def _models_dir(config: dict[str, Any], mode: str) -> Path:
    return data_dir(config) / "models" / mode


def _validation_metrics_path(config: dict[str, Any]) -> Path:
    return data_dir(config) / "reports" / "validation_metrics.json"


def _batch_size(device: torch.device, mode: str) -> int:
    if mode == "smoke":
        return 32
    return 128 if device.type == "cuda" else 32


def _build_dataloaders(config: dict[str, Any], mode: str) -> tuple[RestibleMoveDataset, RestibleMoveDataset]:
    paths = latest_split_paths(config)
    if mode == "smoke":
        train_limit = int(config["training"]["smoke"]["train_positions"])
        val_limit = int(config["training"]["smoke"]["val_positions"])
    else:
        train_limit = 0
        val_limit = 0
    return RestibleMoveDataset(paths["train"], train_limit), RestibleMoveDataset(paths["val"], val_limit)


def _save_checkpoint(
    model: Maia2MoveModel,
    config: dict[str, Any],
    mode: str,
    epoch: int,
    metrics: dict[str, Any],
) -> Path:
    checkpoint_path = _models_dir(config, mode) / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, config, mode, epoch, metrics), checkpoint_path)
    return checkpoint_path


def _train_one_epoch(
    model: Maia2MoveModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    running_loss = 0.0
    batches = 0
    autocast_enabled = device.type == "cuda"

    for boards, labels, elos_self, elos_oppo, _legal_moves, _game_ids, _move_indices in dataloader:
        boards = boards.to(device)
        labels = labels.to(device)
        elos_self = elos_self.to(device)
        elos_oppo = elos_oppo.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            logits = model(boards, elos_self, elos_oppo)
            loss = criterion(logits, labels)

        if autocast_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += float(loss.item())
        batches += 1

    return running_loss / max(1, batches)


def train(config: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode not in {"smoke", "full"}:
        raise ValueError("Training mode must be 'smoke' or 'full'.")

    device = detect_device()
    device_name, gpu_name = device_summary(device)
    print(f"Device: {device_name}")
    print(f"GPU: {gpu_name}")

    train_dataset, val_dataset = _build_dataloaders(config, mode)
    batch_size = _batch_size(device, mode)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_batch_size = 128 if device.type == "cuda" and mode == "full" else 32

    model = load_pretrained_model(device=device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history: dict[str, Any] = {
        "mode": mode,
        "device": device_name,
        "gpu": gpu_name,
        "epochs": [],
    }
    best_checkpoint: Path | None = None
    best_top1 = -1.0

    phase1_epochs = int(config["training"][mode]["phase1_epochs"])
    for epoch in range(1, phase1_epochs + 1):
        model.freeze_body()
        optimizer = AdamW(model.move_head_parameters(), lr=1e-4, weight_decay=1e-4)
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = evaluate_model(model, val_dataset, device=device, split="val", batch_size=val_batch_size)
        write_json(_validation_metrics_path(config), val_metrics)
        history["epochs"].append(
            {"phase": "phase1", "epoch": epoch, "train_loss": train_loss, "val_top1": val_metrics["top1"]["accuracy"]}
        )
        if val_metrics["top1"]["accuracy"] > best_top1:
            best_top1 = val_metrics["top1"]["accuracy"]
            best_checkpoint = _save_checkpoint(model, config, mode, epoch, val_metrics)

    model.unfreeze_all()
    body_parameters = [parameter for parameter in model.body_parameters() if parameter.requires_grad]
    head_parameters = [parameter for parameter in model.move_head_parameters() if parameter.requires_grad]
    optimizer = AdamW(
        [
            {"params": body_parameters, "lr": 1e-5},
            {"params": head_parameters, "lr": 5e-5},
        ],
        weight_decay=1e-4,
    )

    if mode == "smoke":
        phase2_epochs = int(config["training"]["smoke"]["phase2_epochs"])
        patience = None
    else:
        phase2_epochs = int(config["training"]["full"]["phase2_max_epochs"])
        patience = int(config["training"]["full"]["early_stopping_patience"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, phase2_epochs))

    stale_epochs = 0
    for epoch in range(1, phase2_epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = evaluate_model(model, val_dataset, device=device, split="val", batch_size=val_batch_size)
        write_json(_validation_metrics_path(config), val_metrics)
        scheduler.step()
        history["epochs"].append(
            {"phase": "phase2", "epoch": epoch, "train_loss": train_loss, "val_top1": val_metrics["top1"]["accuracy"]}
        )
        if val_metrics["top1"]["accuracy"] > best_top1:
            best_top1 = val_metrics["top1"]["accuracy"]
            stale_epochs = 0
            best_checkpoint = _save_checkpoint(model, config, mode, epoch, val_metrics)
        else:
            stale_epochs += 1
            if patience is not None and stale_epochs >= patience:
                break

    if best_checkpoint is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    history_path = _models_dir(config, mode) / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(history_path, history)

    result = {
        "checkpoint": best_checkpoint,
        "history": history_path,
    }
    if mode == "full":
        result["test_metrics"] = evaluate_checkpoint(config, best_checkpoint, split="test", device=device)
    return result
