## 1. Repository scaffolding

- [x] Create the new repository structure described in `new_implementation.md`.
- [x] Add `.env.example`, `environment.yml`, `pyproject.toml`, `README.md`, and `configs/restible.yaml`.
- [x] Vendor `third_party/maia2` and `third_party/lichess-bot` as pinned submodules.

## 2. Configuration and CLI

- [x] Implement `src/restible_bot/config.py` with YAML loading and `.env` support.
- [x] Implement `src/restible_bot/cli.py` with the required commands for export, dataset prep, training, evaluation, config rendering, UCI, and bot launch.

## 3. Lichess export and dataset preparation

- [x] Implement `src/restible_bot/lichess_export.py` using the pagination fix pattern from `chess_copy`.
- [x] Persist rapid-only PGN exports and metadata under `data/raw/`.
- [x] Implement `src/restible_bot/dataset.py` to filter on `restible`'s own Elo, extract player-only positions, and write prepared game records.
- [x] Split eligible games chronologically by game into train, validation, and test artifacts under `data/splits/`.

## 4. Maia2 integration and training

- [x] Implement `src/restible_bot/maia2_model.py` to load the upstream pretrained `rapid` checkpoint and expose freeze and unfreeze helpers.
- [x] Implement the split-to-tensor conversion layer using Maia2 preprocessing utilities and move vocabulary.
- [x] Implement `src/restible_bot/train.py` with smoke mode and full mode, phase-specific freezing, early stopping, checkpointing, and history output.

## 5. Evaluation and inference

- [x] Implement `src/restible_bot/evaluate.py` to report validation and test metrics with legal-move masking and top-k accuracy.
- [x] Implement `src/restible_bot/inference.py` for deterministic CPU-compatible ranked move inference.

## 6. Bot runtime integration

- [x] Implement `src/restible_bot/uci_engine.py` as a Python UCI adapter that loads the trained Maia2 model once and serves `bestmove`.
- [x] Implement `src/restible_bot/bot_config.py` to render a `lichess-bot` config pointing at the UCI adapter.
- [x] Verify that `LICHESS_BOT_TOKEN` from `.env` is sufficient to render config and launch `third_party/lichess-bot/lichess-bot.py`.

## 7. Acceptance verification

- [x] Verify `export-games` writes all rapid games locally.
- [x] Verify `prepare-dataset` excludes games where `restible` was below `1800`.
- [x] Verify the newest 20% of eligible games become the held-out test split.
- [x] Verify smoke training starts from Maia2 pretrained `rapid` and prints device information.
- [x] Verify phase 1 trains only `last_ln` and `fc_1`, phase 2 unfreezes the full model, and final metrics come from the test split.
- [x] Verify CPU inference, UCI serving, and `lichess-bot` startup work with the fine-tuned checkpoint.
