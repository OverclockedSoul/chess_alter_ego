# Chess alter ego

![alt text](logo.png)

My chess alter ego: a bot that learned my style from 2,000 games using deep learning. Challenge him at https://lichess.org/@/Joan_clone

The bot is based on maia2, another deep learning model that was trained on lots of lichess games (168M) to mimic the human playing style. In this project I used the maia2 final checkpoint to fine-tune a bot on my games to specifically mimic my style.

This same pipeline can be used with any player with a sizable number of games to create a new bot copying that player

## Architecture

At a high level, the system has five parts:

- Lichess export code pulls a player's game history and keeps the rapid games used for training.
- Dataset preparation converts PGN games into move prediction samples with board state, played move, and Elo context.
- A local training wrapper fine-tunes the upstream Maia2 `rapid` model on that dataset using PyTorch.
- Inference code ranks legal moves and can sample from them using different selection policies to make the bot more or less deterministic.
- A UCI bridge connects the trained model to `lichess-bot`, which handles the live Lichess API session and challenge flow.
- For production-style separation, `Dockerfile` serves the FastAPI/model process and `Dockerfile.lichess-bot` runs a lightweight `lichess-bot` process that asks the FastAPI API for moves over HTTP.

## Stack

The main components are:

- `Maia2` for the pretrained chess model backbone
- `PyTorch` for fine-tuning and inference
- `python-chess` for PGN parsing, board handling, and UCI integration
- `lichess-bot` for running the engine on Lichess
- a small Python CLI in `src/restible_bot/` to glue export, training, evaluation, and bot serving together

For installation, retraining, and mode details, see [install.md](/c:/Users/joanc/programming/maia2_chess_clone/install.md).

## Split Lichess Bot Runtime

Start the model-serving API with the existing image:

```sh
docker build -t restible-engine -f Dockerfile .
docker run --rm -p 8001:8001 restible-engine
```

The image defaults to `sample_probability_power_3ply_win_probability` with exponent `2.0`, top-k `3`, and search plies `3`.

Start the separate Lichess runner with the lightweight image:

```sh
docker build -t restible-lichess-bot -f Dockerfile.lichess-bot .
docker run --rm \
  -e LICHESS_BOT_TOKEN="$LICHESS_BOT_TOKEN" \
  -e CHESS_BOT_URL="http://host.docker.internal:8001" \
  restible-lichess-bot
```

`LICHESS_BOT_TOKEN` is required for Lichess. `CHESS_BOT_URL` defaults to `http://127.0.0.1:8001`; set it to the model service URL when the containers are separate. The runner accepts standard casual challenges from humans or bots, with up to 3 parallel games and `clock.limit <= 1800` seconds.
