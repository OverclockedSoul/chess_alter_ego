from __future__ import annotations

import argparse
from pathlib import Path

from .bot_config import render_lichess_bot_config, run_lichess_bot, verify_local_bot_setup
from .config import load_config
from .evaluate import evaluate_checkpoint
from .dataset import prepare_dataset
from .lichess_export import export_games
from .train import train
from .uci_engine import serve_uci


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="restible-bot")
    parser.add_argument("--config", default="configs/restible.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("export-games")
    subparsers.add_parser("prepare-dataset")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--mode", choices=["smoke", "full"], required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--split", choices=["val", "test"], default="test")

    render_parser = subparsers.add_parser("render-bot-config")
    render_parser.add_argument("--checkpoint")
    render_parser.add_argument("--matchmaking", action="store_true")
    render_parser.add_argument("--rated", action="store_true")
    render_parser.add_argument("--initial-time", type=int, default=480)
    render_parser.add_argument("--increment", type=int, default=0)
    render_parser.add_argument("--concurrency", type=int, default=1)

    uci_parser = subparsers.add_parser("run-uci")
    uci_parser.add_argument("--checkpoint", required=True)

    bot_parser = subparsers.add_parser("run-lichess-bot")
    bot_parser.add_argument("--checkpoint")
    bot_parser.add_argument("--max-games", type=int)
    bot_parser.add_argument("--matchmaking", action="store_true")
    bot_parser.add_argument("--rated", action="store_true")
    bot_parser.add_argument("--initial-time", type=int, default=480)
    bot_parser.add_argument("--increment", type=int, default=0)
    bot_parser.add_argument("--concurrency", type=int, default=1)
    bot_parser.add_argument("--poll-interval", type=int, default=30)

    verify_parser = subparsers.add_parser("verify-bot")
    verify_parser.add_argument("--checkpoint")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "export-games":
        result = export_games(config)
        print(result["pgn"])
        print(result["metadata"])
        return

    if args.command == "prepare-dataset":
        result = prepare_dataset(config)
        for path in result.values():
            print(path)
        return

    if args.command == "train":
        result = train(config, args.mode)
        print(result["checkpoint"])
        if "history" in result:
            print(result["history"])
        return

    if args.command == "evaluate":
        metrics = evaluate_checkpoint(config, Path(args.checkpoint), args.split)
        print(metrics)
        return

    if args.command == "render-bot-config":
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        print(
            render_lichess_bot_config(
                config,
                checkpoint,
                allow_matchmaking=args.matchmaking,
                rated=args.rated,
                initial_time=args.initial_time,
                increment=args.increment,
                concurrency=args.concurrency,
            )
        )
        return

    if args.command == "run-uci":
        serve_uci(config["__config_path__"], args.checkpoint)
        return

    if args.command == "run-lichess-bot":
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        result = run_lichess_bot(
            config,
            checkpoint,
            max_games=args.max_games,
            allow_matchmaking=args.matchmaking,
            rated=args.rated,
            initial_time=args.initial_time,
            increment=args.increment,
            concurrency=args.concurrency,
            poll_interval_seconds=args.poll_interval,
        )
        if result is not None:
            print(result)
        return

    if args.command == "verify-bot":
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        print(verify_local_bot_setup(config, checkpoint))
        return


if __name__ == "__main__":
    main()
