## ADDED Requirements

### Requirement: Provide a fresh Maia2-based repository layout

The repository SHALL be reorganized as a new Python project for training and serving a personalized chess bot for `restible`, with source code rooted in `src/restible_bot` and data, configuration, and third-party dependencies separated into stable top-level directories.

#### Scenario: Bootstrap the new repository

- **WHEN** the repository is initialized for the new implementation
- **THEN** it includes `.env.example`, `environment.yml`, `pyproject.toml`, `README.md`, `configs/restible.yaml`, `src/restible_bot/`, `data/`, and `third_party/`
- **AND** it excludes legacy runtime dependencies on `maia-individual`, `lc0`, Docker-only Maia serving, WSL proxies, and TensorFlow inference
- **AND** it vendors upstream `CSSLab/maia2` and `lichess-bot` as pinned submodules

### Requirement: Export all rapid games for restible from Lichess

The system SHALL export all available games for the Lichess user `restible`, paginate until exhaustion, and persist only rapid games and export metadata as local artifacts.

#### Scenario: Export rapid games with pagination

- **WHEN** the user runs `export-games`
- **THEN** the exporter reads `LICHESS_TOKEN` from `.env` when present
- **AND** requests PGN data with moves from `https://lichess.org/api/games/user/restible`
- **AND** paginates using the `until` strategy until no more games are returned
- **AND** classifies rapid games by PGN headers using `Speed == rapid` or an `Event` value containing `rapid`
- **AND** writes a timestamped PGN file and metadata JSON file under `data/raw/`

### Requirement: Prepare a player-specific dataset using restible's own rating

The system SHALL build the personalized corpus from rapid games only, discard games where `restible`'s own player rating is missing or below `1800`, and keep only positions where `restible` is the side to move.

#### Scenario: Filter and extract player moves

- **WHEN** the user runs `prepare-dataset`
- **THEN** the dataset builder detects whether `restible` played White or Black in each game
- **AND** reads `WhiteElo` when `restible` is White or `BlackElo` when `restible` is Black
- **AND** discards the game if that rating is missing or below `1800`
- **AND** iterates only the mainline moves
- **AND** stores per-position samples with `game_id`, `fen`, `move_uci`, `move_index`, `restible_elo`, `opponent_elo`, `color`, `utc_date`, `utc_time`, and `source_pgn`
- **AND** writes game-level records to `data/prepared/restible_games.jsonl`

### Requirement: Split data chronologically by game

The system SHALL split the eligible corpus by game chronology rather than by position so the newest eligible games remain fully held out for evaluation.

#### Scenario: Create train, validation, and test splits

- **WHEN** eligible games are prepared for splitting
- **THEN** the system sorts games by `UTCDate + UTCTime`
- **AND** assigns the newest 20% of eligible games to the test split
- **AND** assigns the remaining 80% to the development pool
- **AND** reserves the newest 10% of the development pool as the validation split
- **AND** writes `train_games.jsonl`, `val_games.jsonl`, `test_games.jsonl`, and `split_summary.json` under `data/splits/`

### Requirement: Fine-tune from the upstream Maia2 rapid checkpoint

The system SHALL start from the upstream Maia2 pretrained `rapid` model, use Maia2 preprocessing utilities and move vocabulary, and fine-tune only for move prediction.

#### Scenario: Load Maia2 and build model inputs

- **WHEN** training or inference initializes the Maia2 wrapper
- **THEN** it calls upstream `from_pretrained(type=\"rapid\", device=...)`
- **AND** unwraps `nn.DataParallel` if needed
- **AND** exposes move-logit-only forward behavior
- **AND** converts split records with Maia2's own board tensorization, move vocabulary, board orientation rules, and Elo bucketing utilities
- **AND** avoids introducing separate tensorization or auxiliary loss paths in v1

### Requirement: Support two-phase fine-tuning with smoke and full modes

The system SHALL train in two phases, first updating only the move head adapter and then fine-tuning the full model, with both smoke and full execution modes.

#### Scenario: Run smoke training

- **WHEN** the user runs `train --mode smoke`
- **THEN** the trainer prints the active device and whether a GPU is available
- **AND** limits training to the first 2000 train positions and first 500 validation positions
- **AND** runs one head-only epoch followed by one full-model epoch
- **AND** uses batch size 32

#### Scenario: Run full training

- **WHEN** the user runs `train --mode full`
- **THEN** phase 1 freezes `chess_cnn`, `to_patch_embedding`, `transformer`, `pos_embedding`, and `elo_embedding`
- **AND** phase 1 updates only `last_ln` and `fc_1`
- **AND** phase 2 unfreezes all parameters and continues optimizing move cross-entropy only
- **AND** phase 2 uses discriminative learning rates with `1e-5` for the body and `5e-5` for the head
- **AND** the optimizer is `AdamW` with weight decay `1e-4` and gradient clipping `1.0`
- **AND** full training runs one warmup epoch, up to eight fine-tuning epochs, cosine decay, validation early stopping with patience two, and best-checkpoint persistence
- **AND** writes checkpoints and history under `data/models/`

### Requirement: Report held-out evaluation metrics

The system SHALL evaluate the best checkpoint on held-out data using legal-move masking and persist machine-readable and human-readable reports.

#### Scenario: Evaluate validation or test splits

- **WHEN** the user runs evaluation during training or against a saved checkpoint
- **THEN** the evaluator computes game count, position count, top-1 accuracy, top-3 accuracy, top-5 accuracy, and mean probability assigned to the played move
- **AND** each top-k metric includes exact numerator and denominator counts
- **AND** reported probabilities are computed after masking to legal moves and applying softmax
- **AND** validation metrics are written to `data/reports/validation_metrics.json`
- **AND** final test metrics are written to `data/reports/test_metrics.json` and `data/reports/test_metrics.md`
- **AND** final success reporting is based on the held-out test split rather than training accuracy

### Requirement: Provide deterministic CPU-compatible inference

The system SHALL support inference on CPU without changing model weights and select the highest-probability legal move deterministically.

#### Scenario: Rank legal moves for a position

- **WHEN** inference is requested for a FEN and Elo context
- **THEN** the inference module loads the fine-tuned checkpoint through the Maia2 wrapper
- **AND** accepts `fen`, `elo_self`, and `elo_oppo`
- **AND** masks illegal moves before ranking
- **AND** returns ranked legal moves with probabilities
- **AND** defaults to `elo_self = 1900` and `elo_oppo = 1900` when Lichess opponent rating is unavailable
- **AND** chooses the top-probability legal move without stochastic sampling

### Requirement: Integrate with lichess-bot through a Python UCI adapter

The system SHALL provide a Python UCI bridge and generate a `lichess-bot` configuration that launches it from the repository root.

#### Scenario: Serve UCI commands with the trained model

- **WHEN** `src/restible_bot/uci_engine.py` is started
- **THEN** it loads the fine-tuned model once at process startup
- **AND** handles `uci`, `isready`, `ucinewgame`, `position`, `go`, and `quit`
- **AND** on `go` prints an `info` line followed by `bestmove <uci>`

#### Scenario: Render and run a lichess-bot configuration

- **WHEN** the user runs `render-bot-config` or `run-lichess-bot`
- **THEN** the bot configuration loader reads `.env`
- **AND** uses `LICHESS_BOT_TOKEN` as the bot credential source
- **AND** writes `data/artifacts/lichess-bot/config.yml`
- **AND** configures the engine as a UCI process that launches the Python executable against `src/restible_bot/uci_engine.py`
- **AND** launches upstream `third_party/lichess-bot/lichess-bot.py` with the generated config when requested

### Requirement: Expose the required command-line interface

The system SHALL provide a CLI entry point that exposes the export, dataset preparation, training, evaluation, UCI, and `lichess-bot` workflows required by the repository.

#### Scenario: Invoke supported commands

- **WHEN** the CLI is installed or executed as the project entry point
- **THEN** it exposes `export-games`, `prepare-dataset`, `train --mode smoke`, `train --mode full`, `evaluate --checkpoint <path> --split test`, `render-bot-config`, `run-uci`, and `run-lichess-bot`
