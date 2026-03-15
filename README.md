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

## Stack

The main components are:

- `Maia2` for the pretrained chess model backbone
- `PyTorch` for fine-tuning and inference
- `python-chess` for PGN parsing, board handling, and UCI integration
- `lichess-bot` for running the engine on Lichess
- a small Python CLI in `src/restible_bot/` to glue export, training, evaluation, and bot serving together

For installation, retraining, and mode details, see [install.md](/c:/Users/joanc/programming/maia2_chess_clone/install.md).
