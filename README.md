# restible-bot

Fresh Maia2-based training and serving pipeline for a personalized Lichess bot modeled on `restible`.

## Layout

- `configs/restible.yaml`: project configuration
- `src/restible_bot/`: application code
- `data/`: raw exports, prepared datasets, splits, models, reports, and bot artifacts
- `third_party/maia2/`: upstream Maia2 submodule
- `third_party/lichess-bot/`: upstream lichess-bot submodule

## Setup

```bash
conda env create -f environment.yml
conda activate chess_clone
pip install -e .
```

Copy `.env.example` to `.env` and provide:

```bash
LICHESS_TOKEN=...
LICHESS_BOT_TOKEN=...
```

## Commands

```bash
restible-bot export-games
restible-bot prepare-dataset
restible-bot train --mode smoke
restible-bot train --mode full
restible-bot evaluate --checkpoint data/models/full/best.pt --split test
restible-bot render-bot-config
restible-bot run-uci --checkpoint data/models/full/best.pt
restible-bot run-lichess-bot --checkpoint data/models/full/best.pt
```

## Notes

- The training pipeline starts from the upstream Maia2 pretrained `rapid` checkpoint.
- Dataset filtering is based on `restible`'s own game rating, not average game Elo.
- Splits are chronological by game so the newest eligible games stay held out.
- Inference masks illegal moves and is CPU-compatible.
