from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn


def _ensure_maia2_path() -> None:
    maia2_root = Path(__file__).resolve().parents[2] / "third_party" / "maia2"
    if str(maia2_root) not in sys.path:
        sys.path.insert(0, str(maia2_root))


class Maia2MoveModel(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, boards: torch.Tensor, elos_self: torch.Tensor, elos_oppo: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.backbone(boards, elos_self, elos_oppo)
        return logits

    def freeze_body(self) -> None:
        for module_name in ["chess_cnn", "to_patch_embedding", "transformer", "pos_embedding", "elo_embedding"]:
            module = getattr(self.backbone, module_name)
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            else:
                for parameter in module.parameters():
                    parameter.requires_grad = False
        self.freeze_auxiliary_heads()
        for module_name in ["last_ln", "fc_1"]:
            module = getattr(self.backbone, module_name)
            for parameter in module.parameters():
                parameter.requires_grad = True

    def freeze_auxiliary_heads(self) -> None:
        for module_name in ["fc_2", "fc_3", "fc_3_1"]:
            module = getattr(self.backbone, module_name)
            for parameter in module.parameters():
                parameter.requires_grad = False

    def unfreeze_all(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def move_head_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        for module_name in ["last_ln", "fc_1"]:
            parameters.extend(list(getattr(self.backbone, module_name).parameters()))
        return parameters

    def body_parameters(self) -> list[nn.Parameter]:
        head_ids = {id(parameter) for parameter in self.move_head_parameters()}
        return [parameter for parameter in self.backbone.parameters() if id(parameter) not in head_ids]


def load_pretrained_model(device: torch.device, save_root: str | Path | None = None) -> Maia2MoveModel:
    _ensure_maia2_path()
    from maia2.model import from_pretrained

    save_root = save_root or (Path(__file__).resolve().parents[2] / "data" / "models" / "pretrained")
    upstream_device = "gpu" if device.type == "cuda" else "cpu"
    backbone = from_pretrained(type="rapid", device=upstream_device, save_root=str(save_root))
    if isinstance(backbone, nn.DataParallel):
        backbone = backbone.module
    return Maia2MoveModel(backbone).to(device)


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
    save_root: str | Path | None = None,
) -> tuple[Maia2MoveModel, dict[str, Any]]:
    model = load_pretrained_model(device=device, save_root=save_root)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    return model, checkpoint


def checkpoint_payload(
    model: Maia2MoveModel,
    config: dict[str, Any],
    mode: str,
    epoch: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "mode": mode,
        "epoch": epoch,
        "metrics": metrics,
        "config_path": config["__config_path__"],
    }
