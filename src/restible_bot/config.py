from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml

from .utils import ensure_dir


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "root": ".",
        "data_dir": "data",
    },
    "lichess": {
        "username": "restible",
        "base_url": "https://lichess.org",
        "token_env": "LICHESS_TOKEN",
        "request_batch_size": 300,
    },
    "dataset": {
        "speed": "rapid",
        "min_player_elo": 1800,
        "test_fraction": 0.20,
        "validation_fraction_within_train": 0.10,
        "split_seed": 42,
    },
    "training": {
        "target_self_elo": 1900,
        "output_suffix": "",
        "override_self_elo": None,
        "override_opponent_elo": None,
        "smoke": {
            "train_positions": 2000,
            "val_positions": 500,
            "phase1_epochs": 1,
            "phase2_epochs": 1,
        },
        "full": {
            "phase1_epochs": 1,
            "phase2_max_epochs": 8,
            "early_stopping_patience": 2,
        },
    },
    "bot": {
        "token_env": "LICHESS_BOT_TOKEN",
        "config_output": "data/artifacts/lichess-bot/config.yml",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path = "configs/restible.yaml") -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    user_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = _deep_merge(DEFAULT_CONFIG, user_config)
    root_value = str(config["project"]["root"])
    root_path = (config_path.parent / root_value).resolve()
    if root_value == "." and config_path.parent.name == "configs":
        root_path = config_path.parent.parent.resolve()
    config["project"]["root"] = str(root_path)
    config["__config_path__"] = str(config_path)
    load_dotenv(root_path / ".env", override=False)
    ensure_project_layout(config)
    return config


def project_root(config: dict[str, Any]) -> Path:
    return Path(config["project"]["root"]).resolve()


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root(config) / path


def data_dir(config: dict[str, Any]) -> Path:
    return ensure_dir(resolve_path(config, config["project"]["data_dir"]))


def ensure_project_layout(config: dict[str, Any]) -> None:
    root = project_root(config)
    ensure_dir(root / "configs")
    ensure_dir(root / "src" / "restible_bot")
    for relative in [
        "data/raw",
        "data/prepared",
        "data/splits",
        "data/models",
        "data/reports",
        "data/artifacts/lichess-bot",
    ]:
        ensure_dir(root / relative)
