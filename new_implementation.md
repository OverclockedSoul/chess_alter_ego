# New Bot Implementation From Scratch

## Goal

Build a fresh repository that trains a personalized chess bot for `restible` using `CSSLab/maia2` https://github.com/CSSLab/maia2 as the base model, then connects that model to Lichess through `lichess-bot`.

The implementation must do this:

1. Export all `restible` games from Lichess.
2. Keep only rapid games.
3. Keep only games where `restible`'s own game rating is `>= 1800`.
4. Split eligible games by game, not by position:
   - newest 20% of eligible games = test set
   - remaining 80% = development pool
   - from the development pool, reserve the newest 10% as validation
5. Start from the upstream `maia2` pretrained `rapid` model.
6. Fine-tune in two phases:
   - phase 1: freeze the model body, train only the move head adapter
   - phase 2: unfreeze the full model and continue fine-tuning
7. Report final metrics on the held-out 20% test set.
8. Run the model behind `lichess-bot` with CPU-compatible inference.

This document is written so another LLM or engineer can implement the new repo without inheriting the broken `maia-individual` / `lc0` path.

## Hard Decisions

### Use `maia2`

### Start from `maia2` pretrained `rapid`, not a 1900 checkpoint

Upstream `maia2` exposes pretrained models by time control, not by Elo bucket. The correct starting point is the pretrained `rapid` model.

### Filter on `restible`'s rating, not average game rating

Use:

- `WhiteElo` if `restible` is White
- `BlackElo` if `restible` is Black

Discard the game if that rating is below `1800` or missing.

This is stricter and more faithful than filtering on average game Elo.

### Split by game chronologically

Do not randomly split positions. That leaks opening structure and repeated positions across train and test.

Use chronological split after filtering:

1. parse all eligible games
2. sort by `UTCDate + UTCTime`
3. assign:
   - first 80% -> train
   - next 10% -> validation
   - last 10% -> test

### Optimize for move prediction only

The personalized bot objective is to predict `restible`'s move choices on unseen `restible` rapid games.

For fine-tuning, optimize the move policy head only. Do not spend time reproducing Maia2's full multi-task training unless needed later.

That means:

- warm-start from pretrained `maia2`
- use the upstream feature extractor
- fine-tune against cross-entropy on the actual played move
- report move prediction metrics on validation and test

## New Repository Layout

Use this structure:

```text
<new-repo>/
  .env.example
  environment.yml
  pyproject.toml
  README.md
  configs/
    restible.yaml
  data/
    raw/
    prepared/
    splits/
    models/
    reports/
  third_party/
    maia2/
    lichess-bot/
  src/
    restible_bot/
      __init__.py
      config.py
      lichess_export.py
      dataset.py
      maia2_model.py
      train.py
      evaluate.py
      inference.py
      uci_engine.py
      bot_config.py
      cli.py
```

## Dependencies

Use a new conda env named `chess_clone`.

Recommended stack:

- Python 3.11
- PyTorch with CUDA for training
- `python-chess`
- `requests`
- `PyYAML`
- `pandas`
- `numpy`
- `tqdm`
- `python-dotenv`
- `scikit-learn` only if needed for convenience, not required

Do not include:

- `lc0`
- `trainingdata-tool`
- `maia-individual`
- TensorFlow

Vendor these upstream repos as pinned submodules:

- `https://github.com/CSSLab/maia2`
- `https://github.com/lichess-bot-devs/lichess-bot`

## What To Reuse From The Current Repo

Copy logic from these absolute paths.

### Reuse directly or adapt heavily

- `C:\Users\joanc\programming\chess_copy\src\chess_clone\dataset.py`
  - reusable: Lichess export pagination, PGN parsing, rapid filtering, per-game iteration
  - do not keep its current rating filter semantics
- `C:\Users\joanc\programming\chess_copy\src\chess_clone\config.py`
  - reusable: YAML config loading and `.env` loading
- `C:\Users\joanc\programming\chess_copy\src\chess_clone\bot.py`
  - reusable: `lichess-bot` config rendering pattern and token loading
  - do not keep the Docker `lc0` engine payload
- `C:\Users\joanc\programming\chess_copy\docker\maia\evaluate_metrics.py`
  - reusable: simple JSON metric report pattern
- `C:\Users\joanc\programming\chess_copy\third_party\lichess-bot\wiki\Configure-lichess-bot.md`
- `C:\Users\joanc\programming\chess_copy\third_party\lichess-bot\wiki\Create-a-homemade-engine.md`
- `C:\Users\joanc\programming\chess_copy\third_party\lichess-bot\lichess-bot.py`

### Do not reuse

- `C:\Users\joanc\programming\chess_copy\src\chess_clone\maia.py`
- `C:\Users\joanc\programming\chess_copy\src\chess_clone\docker_lc0_proxy.py`
- `C:\Users\joanc\programming\chess_copy\src\chess_clone\wsl_lc0_proxy.py`
- `C:\Users\joanc\programming\chess_copy\docker\maia\*`
- `C:\Users\joanc\programming\chess_copy\third_party\maia-individual\*`
- `C:\Users\joanc\programming\chess_copy\third_party\trainingdata-tool\*`

## Data Pipeline

### 1. Export from Lichess

Implement `src/restible_bot/lichess_export.py`.

Behavior:

- read `LICHESS_TOKEN` from `.env` if present
- call `https://lichess.org/api/games/user/restible`
- fetch all games using pagination via `until`
- request PGN with moves
- do not rely on Lichess `perfType` filtering
- filter rapid locally from PGN headers

Rapid detection:

- `Speed == rapid`, or
- `"rapid"` appears in `Event`

Persist:

- `data/raw/restible_all_rapid_<timestamp>.pgn`
- `data/raw/restible_all_rapid_<timestamp>.metadata.json`

The exporter should preserve the current pagination fix from:

- `C:\Users\joanc\programming\chess_copy\src\chess_clone\dataset.py`

### 2. Build the personal training corpus

Implement `src/restible_bot/dataset.py`.

For each rapid game:

1. detect whether `restible` played White or Black
2. read `restible`'s own Elo from the corresponding PGN header
3. discard the game if `restible`'s Elo is missing or `< 1800`
4. iterate through the mainline
5. keep only positions where the side to move is `restible`

For each sample, store:

- `game_id`
- `fen`
- `move_uci`
- `move_index`
- `restible_elo`
- `opponent_elo`
- `color`
- `utc_date`
- `utc_time`
- `source_pgn`

Store one game per record first, then split by game. Do not flatten into positions before splitting.

Persist:

- `data/prepared/restible_games.jsonl`
- `data/splits/train_games.jsonl`
- `data/splits/val_games.jsonl`
- `data/splits/test_games.jsonl`
- `data/splits/split_summary.json`

### 3. Convert to `maia2` fine-tuning records

Implement a conversion layer that transforms each player move into the inputs needed by `maia2`.

Use `maia2`'s own board preprocessing utilities rather than reimplementing a different tensorization scheme.

Each sample passed into the trainer must produce:

- board tensor
- move index in `all_moves_dict`
- elo bucket for `restible`
- elo bucket for opponent

Important:

- keep the board orientation conventions consistent with upstream `maia2`
- if upstream mirrors black positions, follow that exactly
- use the same move vocabulary as upstream

The safest approach is:

1. import `board_to_tensor`, `create_elo_dict`, and `map_to_category` from `third_party/maia2`
2. implement a small dataset wrapper in our repo that converts the split JSONL into tensors

## Model Implementation

### Upstream model facts to rely on

The `maia2` model body is approximately:

- `chess_cnn`
- `to_patch_embedding`
- `transformer`
- `pos_embedding`
- `elo_embedding`

The move prediction head is:

- `last_ln`
- `fc_1`

There are additional upstream auxiliary heads:

- `fc_2`
- `fc_3`
- `fc_3_1`

For this new repo, fine-tune only the move policy path unless a later experiment proves auxiliary losses help.

### 4. Implement a thin model wrapper

Implement `src/restible_bot/maia2_model.py`.

Responsibilities:

1. load the upstream pretrained `rapid` model
2. expose helpers to freeze and unfreeze modules
3. expose a forward method that returns move logits only
4. expose a helper to build an inference-ready model on CPU or GPU

Implementation detail:

- call upstream `from_pretrained(type="rapid", device=...)`
- immediately unwrap `nn.DataParallel` if upstream returns it wrapped
- keep the original model weights format compatible with `state_dict`

### 5. Freeze policy for phase 1

In phase 1, freeze:

- `chess_cnn`
- `to_patch_embedding`
- `transformer`
- `pos_embedding`
- `elo_embedding`

Train only:

- `last_ln`
- `fc_1`

Leave `fc_2`, `fc_3`, and `fc_3_1` frozen and unused.

### 6. Full fine-tuning for phase 2

In phase 2:

- unfreeze all parameters
- continue optimizing move cross-entropy only

Use discriminative learning rates:

- body lr: `1e-5`
- head lr: `5e-5`

Use:

- optimizer: `AdamW`
- weight decay: `1e-4`
- gradient clipping: `1.0`
- mixed precision on GPU if available

## Training Procedure

Implement `src/restible_bot/train.py`.

### Smoke mode

Keep a smoke mode for fast validation before the full run.

Smoke mode rules:

- use only the first `2000` train positions and first `500` validation positions
- phase 1: `1` epoch
- phase 2: `1` epoch
- batch size: `32`
- print the device at startup
- explicitly print whether CUDA is available and which GPU is being used

This preserves the earlier requirement to verify the pipeline quickly before full training.

### Full mode

Full mode rules:

- phase 1: `1` epoch head-only warmup
- phase 2: up to `8` epochs full fine-tuning
- early stopping on validation top-1 accuracy with patience `2`
- keep the best checkpoint by validation top-1 accuracy

Batch size:

- default `128` on GPU
- fallback `32` on CPU

Learning rate schedule:

- phase 1: constant lr `1e-4` for `last_ln` and `fc_1`
- phase 2: cosine decay

Artifacts:

- `data/models/smoke/best.pt`
- `data/models/full/best.pt`
- `data/models/full/training_history.json`

### Loss

Use only move cross-entropy.

Do not include side-info or value losses in v1. The purpose of this repo is to clone `restible`'s move choices, and move accuracy is the required metric.

## Evaluation

Implement `src/restible_bot/evaluate.py`.

Do not report training accuracy as the final metric. Report held-out test metrics only.

### Evaluation datasets

Run evaluation on:

- validation split during training
- test split once at the end using the best validation checkpoint

### Required metrics

At minimum compute:

- number of test games
- number of test positions
- top-1 accuracy
- top-3 accuracy
- top-5 accuracy
- mean probability assigned to the actual played move
- exact numerator and denominator counts for top-1, top-3, top-5

Report format example:

```json
{
  "split": "test",
  "games": 123,
  "positions": 4567,
  "top1": {
    "correct": 1234,
    "total": 4567,
    "accuracy": 0.2702
  },
  "top3": {
    "correct": 2311,
    "total": 4567,
    "accuracy": 0.5061
  },
  "top5": {
    "correct": 2789,
    "total": 4567,
    "accuracy": 0.6107
  },
  "mean_true_move_probability": 0.2144
}
```

Persist:

- `data/reports/validation_metrics.json`
- `data/reports/test_metrics.json`
- `data/reports/test_metrics.md`

### Exact metric definitions

- `top-1`: true move is the highest-probability legal move
- `top-3`: true move is in the 3 highest-probability legal moves
- `top-5`: true move is in the 5 highest-probability legal moves
- `mean_true_move_probability`: average model probability assigned to the actual played move after masking to legal moves and applying softmax

### Optional, but recommended

Also report metrics by phase of game:

- opening: plies `0-15`
- middlegame: plies `16-39`
- late game: plies `40+`

This is not required for acceptance, but it is useful.

## Inference

Implement `src/restible_bot/inference.py`.

Requirements:

- inference must run on CPU
- inference may use GPU if available, but CPU must work without changing model weights
- load the saved fine-tuned checkpoint into the `maia2` wrapper
- take `fen`, `elo_self`, `elo_oppo`
- mask illegal moves
- return a ranked move list and probabilities

Default inference policy for the bot:

- choose the legal move with highest probability

Do not add stochastic sampling in v1. Keep behavior deterministic until the base pipeline is stable.

For Lichess games:

- `elo_self = 1900`
- `elo_oppo = opponent rating if known, else 1900`

## UCI Bridge For `lichess-bot`

Implement `src/restible_bot/uci_engine.py`.

This should be a small Python UCI adapter, not `lc0`.

Responsibilities:

- load the fine-tuned model once at process startup
- answer standard UCI commands:
  - `uci`
  - `isready`
  - `ucinewgame`
  - `position`
  - `go`
  - `quit`
- on `go`, compute the best move from the current board and print:
  - an `info` line
  - `bestmove <uci>`

Reuse the command parsing style from:

- `C:\Users\joanc\programming\chess_copy\docker\maia\uci_service.py`

But replace the old direct Maia-individual inference with the new `maia2` inference module.

## `lichess-bot` Integration

Implement `src/restible_bot/bot_config.py`.

Responsibilities:

- load `.env`
- read `LICHESS_BOT_TOKEN`
- generate a `lichess-bot` config YAML pointing to the UCI adapter

Base the YAML shape on:

- `C:\Users\joanc\programming\chess_copy\src\chess_clone\bot.py`

But the engine entry must point to the new Python UCI adapter instead of any Docker or `lc0` launcher.

Expected engine settings:

- `protocol: uci`
- working directory = repo root
- command = Python executable launching `src/restible_bot/uci_engine.py`

Store generated config at:

- `data/artifacts/lichess-bot/config.yml`

Then launch upstream bot with:

- `python third_party/lichess-bot/lichess-bot.py --config data/artifacts/lichess-bot/config.yml`

## CLI Surface

Implement `src/restible_bot/cli.py`.

Required commands:

- `export-games`
- `prepare-dataset`
- `train --mode smoke`
- `train --mode full`
- `evaluate --checkpoint <path> --split test`
- `render-bot-config`
- `run-uci`
- `run-lichess-bot`

The training command must print the startup device line, for example:

```text
Device: cuda
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

or:

```text
Device: cpu
GPU: not available
```

## Config

Implement `configs/restible.yaml`.

Required fields:

```yaml
project:
  root: .
  data_dir: data

lichess:
  username: restible
  base_url: https://lichess.org
  token_env: LICHESS_TOKEN

dataset:
  speed: rapid
  min_player_elo: 1800
  test_fraction: 0.20
  validation_fraction_within_train: 0.10

training:
  target_self_elo: 1900
  smoke:
    train_positions: 2000
    val_positions: 500
    phase1_epochs: 1
    phase2_epochs: 1
  full:
    phase1_epochs: 1
    phase2_max_epochs: 8
    early_stopping_patience: 2

bot:
  token_env: LICHESS_BOT_TOKEN
  config_output: data/artifacts/lichess-bot/config.yml
```

## Acceptance Criteria

The new repo is complete only if all of the following are true:

1. `export-games` fetches all available `restible` games and writes all rapid games locally.
2. `prepare-dataset` filters out games where `restible` was below `1800`.
3. The split is by game and yields exactly 20% test games.
4. The trainer starts from upstream `maia2` pretrained `rapid`.
5. Phase 1 freezes the body and updates only `last_ln` and `fc_1`.
6. Phase 2 unfreezes the whole model and fine-tunes end-to-end.
7. Final reported metrics are on the held-out test set, not the training set.
8. Inference works on CPU.
9. The generated UCI engine works with `lichess-bot`.
10. `LICHESS_BOT_TOKEN` in `.env` is enough to start the local bot process.

## Suggested Build Order

Implement in this order:

1. config loading and `.env` handling
2. full Lichess export with pagination
3. rapid-only and `restible >= 1800` filtering
4. chronological game-level split
5. `maia2` wrapper and pretrained `rapid` loading
6. smoke training
7. full training with validation checkpointing
8. test-set evaluation report
9. CPU inference
10. UCI adapter
11. `lichess-bot` config generation
12. local bot run

## Explicit Non-Goals

Do not implement any of these in the fresh repo:

- `maia-individual`
- `train_transfer.py`
- `lc0`
- WSL-specific runtime proxies
- Docker-based Maia runtime
- TensorFlow-based inference
- benchmark Elo estimation against heuristic opponents

The first deliverable is a personalized `maia2` move-prediction bot with test-set accuracy metrics and Lichess connectivity.
