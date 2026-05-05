# Installation and Retraining

This repository installs a local training and serving pipeline for a Maia2-based Lichess bot. The commands below use the default configuration in `configs/restible.yaml`.

## Prerequisites

- Conda
- Python 3.11
- A Lichess account token for exporting games: `LICHESS_TOKEN`
- A Lichess bot token for running the bot: `LICHESS_BOT_TOKEN`

If this is a fresh clone, make sure the submodules are present:

```bash
git submodule update --init --recursive
```

## Install

Create the environment and install the package:

```bash
conda env create -f environment.yml
conda activate chess_clone
pip install -e .
```

Create `.env` from the example and add your tokens:

```bash
copy .env.example .env
```

```dotenv
LICHESS_TOKEN=your_export_token
LICHESS_BOT_TOKEN=your_bot_token
```

## Configure the Target Player

Edit `configs/restible.yaml` before exporting or retraining if you want to clone a different player or change filtering:

- `lichess.username`: Lichess username to clone
- `dataset.speed`: currently expected to be `rapid`
- `dataset.min_player_elo`: minimum rating for the target player's own games
- `training.smoke` and `training.full`: training loop sizes and epoch counts

## Export Lichess Data

Fetch the target player's games from Lichess and write a rapid-only PGN export under `data/raw/`:

```bash
restible-bot export-games
```

This produces:

- a PGN file such as `data/raw/restible_all_rapid_<timestamp>.pgn`
- a matching metadata file with export details

## Prepare the Dataset

Convert the PGN export into train/validation/test artifacts:

```bash
restible-bot prepare-dataset
```

This writes:

- `data/prepared/restible_games.jsonl`
- `data/splits/train_games.jsonl`
- `data/splits/val_games.jsonl`
- `data/splits/test_games.jsonl`
- `data/splits/split_summary.json`

## Retrain the Bot

Training starts from the upstream Maia2 `rapid` checkpoint.

Use `smoke` for a quick pipeline check:

```bash
restible-bot train --mode smoke
```

Use `full` for the real fine-tuning run:

```bash
restible-bot train --mode full
```

Outputs are written under `data/models/smoke/` or `data/models/full/`, including `best.pt`. Full training also evaluates the saved checkpoint on the test split.

## Evaluate a Checkpoint

```bash
restible-bot evaluate --checkpoint data/models/full/best.pt --split test
```

Use `--split val` to inspect validation performance instead.

## Run the Model Locally

Start the trained model as a UCI engine:

```bash
restible-bot run-uci --checkpoint data/models/full/best.pt
```

Check that the local bot setup is valid:

```bash
restible-bot verify-bot --checkpoint data/models/full/best.pt
```

Render a `lichess-bot` config without launching games:

```bash
restible-bot render-bot-config --checkpoint data/models/full/best.pt
```

## Run on Lichess

Launch the bot with the default checkpoint:

```bash
restible-bot run-lichess-bot --checkpoint data/models/full/best.pt
```

Useful variants:

```bash
restible-bot run-lichess-bot --checkpoint data/models/full/best.pt --matchmaking --rated
restible-bot run-lichess-bot --checkpoint data/models/full/best.pt --challenge-user some_user --max-games 10
restible-bot run-lichess-bot --checkpoint data/models/full/best.pt --run-hours 2
```

## Run Split Containers

The default `Dockerfile` remains the FastAPI/model-serving image. It loads the PyTorch model and exposes `/move`:

```bash
docker build -t restible-engine -f Dockerfile .
docker run --rm -p 8001:8001 restible-engine
```

The model server defaults to `CHESS_BOT_SELECTION_POLICY=sample_probability_power_3ply_win_probability`, `CHESS_BOT_PROBABILITY_EXPONENT=2.0`, `CHESS_BOT_SEARCH_TOP_K=3`, and `CHESS_BOT_SEARCH_PLIES=3`.

`Dockerfile.lichess-bot` is the lightweight Lichess runner. It does not start FastAPI and its UCI adapter calls `POST {CHESS_BOT_URL}/move` with `{ "fen": board.fen() }`.

```bash
docker build -t restible-lichess-bot -f Dockerfile.lichess-bot .
docker run --rm \
  -e LICHESS_BOT_TOKEN="$LICHESS_BOT_TOKEN" \
  -e CHESS_BOT_URL="http://host.docker.internal:8001" \
  restible-lichess-bot
```

Required env:

- `LICHESS_BOT_TOKEN`: Lichess bot OAuth token.
- `CHESS_BOT_URL`: model API base URL. Defaults to `http://127.0.0.1:8001`.

The split runner accepts incoming standard casual challenges from humans or bots, allows at most 3 parallel games, and enforces `clock.limit <= 1800` seconds through `challenge.max_base`.

## Move-Selection Modes

The training modes are only `smoke` and `full`. The play-style modes such as `p^2` and `p*win` are exposed as inference `--selection-policy` options.

Available policies:

- `top1`: always play the highest-probability move
- `sample_top2`: sample between the top 2 moves
- `sample_top3`: sample between the top 3 moves
- `sample_min_probability`: sample from moves above `--min-probability`
- `sample_reweighted_below_threshold`: downweight low-probability moves
- `sample_probability_power`: sample with weights proportional to `p^k`
- `sample_probability_times_win_probability`: sample with weights proportional to `p * win_prob`

Examples:

`p^2` style move selection:

```bash
restible-bot run-uci --checkpoint data/models/full/best.pt --selection-policy sample_probability_power --probability-exponent 2.0
```

`p^3` style move selection:

```bash
restible-bot run-uci --checkpoint data/models/full/best.pt --selection-policy sample_probability_power --probability-exponent 3.0
```

`p*win` style move selection:

```bash
restible-bot run-uci --checkpoint data/models/full/best.pt --selection-policy sample_probability_times_win_probability
```

The same flags also work with `render-bot-config` and `run-lichess-bot`.
