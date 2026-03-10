## 1. Repository scaffolding

- [ ] Create the new repository structure described in `new_implementation.md`.
- [ ] Add `.env.example`, `environment.yml`, `pyproject.toml`, `README.md`, and `configs/restible.yaml`.
- [ ] Vendor `third_party/maia2` and `third_party/lichess-bot` as pinned submodules.

## 2. Configuration and CLI

- [ ] Implement `src/restible_bot/config.py` with YAML loading and `.env` support.
- [ ] Implement `src/restible_bot/cli.py` with the required commands for export, dataset prep, training, evaluation, config rendering, UCI, and bot launch.

## 3. Lichess export and dataset preparation

- [ ] Implement `src/restible_bot/lichess_export.py` using the pagination fix pattern from `chess_copy`.
- [ ] Persist rapid-only PGN exports and metadata under `data/raw/`.
- [ ] Implement `src/restible_bot/dataset.py` to filter on `restible`'s own Elo, extract player-only positions, and write prepared game records.
- [ ] Split eligible games chronologically by game into train, validation, and test artifacts under `data/splits/`.

## 4. Maia2 integration and training

- [ ] Implement `src/restible_bot/maia2_model.py` to load the upstream pretrained `rapid` checkpoint and expose freeze and unfreeze helpers.
- [ ] Implement the split-to-tensor conversion layer using Maia2 preprocessing utilities and move vocabulary.
- [ ] Implement `src/restible_bot/train.py` with smoke mode and full mode, phase-specific freezing, early stopping, checkpointing, and history output.

## 5. Evaluation and inference

- [ ] Implement `src/restible_bot/evaluate.py` to report validation and test metrics with legal-move masking and top-k accuracy.
- [ ] Implement `src/restible_bot/inference.py` for deterministic CPU-compatible ranked move inference.

## 6. Bot runtime integration

- [ ] Implement `src/restible_bot/uci_engine.py` as a Python UCI adapter that loads the trained Maia2 model once and serves `bestmove`.
- [ ] Implement `src/restible_bot/bot_config.py` to render a `lichess-bot` config pointing at the UCI adapter.
- [ ] Verify that `LICHESS_BOT_TOKEN` from `.env` is sufficient to render config and launch `third_party/lichess-bot/lichess-bot.py`.

## 7. Acceptance verification

- [ ] Verify `export-games` writes all rapid games locally.
- [ ] Verify `prepare-dataset` excludes games where `restible` was below `1800`.
- [ ] Verify the newest 20% of eligible games become the held-out test split.
- [ ] Verify smoke training starts from Maia2 pretrained `rapid` and prints device information.
- [ ] Verify phase 1 trains only `last_ln` and `fc_1`, phase 2 unfreezes the full model, and final metrics come from the test split.
- [ ] Verify CPU inference, UCI serving, and `lichess-bot` startup work with the fine-tuned checkpoint.
