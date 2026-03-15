# Proposal: Reimplement the restible bot around Maia2

## Why

The current bot implementation path in `C:\Users\joanc\programming\chess_copy` mixes Maia-related code with `maia-individual`, `lc0`, Docker-specific runtime pieces, and evaluation shortcuts that do not match the desired end state. This change defines a fresh repository implementation that keeps the useful export and integration patterns while replacing the broken model and serving path with a clean Maia2-based training and inference stack.

## What Changes

- Create a new repository layout centered on `src/restible_bot`.
- Export all `restible` games from Lichess and keep only rapid games.
- Filter games using `restible`'s own rating in the game and discard games below `1800`.
- Split eligible games chronologically by game, not by position.
- Start from the upstream Maia2 pretrained `rapid` checkpoint.
- Fine-tune in two phases: head-only warmup, then full-model tuning.
- Report held-out test metrics from the newest 20% of eligible games.
- Provide CPU-compatible inference, a Python UCI bridge, and `lichess-bot` integration.
- Reuse only the pieces of `chess_copy` that are still valid for config loading, Lichess export pagination, metric reporting shape, and `lichess-bot` config generation patterns.

## Non-Goals

- Reusing `maia-individual`.
- Reusing `lc0` or any Docker/WSL runtime proxy.
- Reproducing Maia2's auxiliary training objectives in v1.
- Adding stochastic move sampling in the first bot release.
- Building TensorFlow-based inference or legacy transfer-training scripts.

## Impact

- Establishes a new OpenSpec capability for the personalized `restible` Maia2 bot.
- Replaces the old runtime assumptions with a single Python-first implementation.
- Defines the acceptance criteria for the new repository before implementation starts.
