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

SELECTION_POLICIES = [
    "top1",
    "sample_top2",
    "sample_top3",
    "sample_min_probability",
    "sample_reweighted_below_threshold",
    "sample_probability_power",
    "sample_probability_times_win_probability",
]


def _add_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection-policy", choices=SELECTION_POLICIES, default="sample_probability_power")
    parser.add_argument("--min-probability", type=float, default=0.20)
    parser.add_argument("--below-threshold-weight-scale", type=float, default=0.25)
    parser.add_argument("--probability-exponent", type=float, default=2.0)


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
    render_parser.add_argument("--opponent-min-rating", type=int)
    render_parser.add_argument("--opponent-max-rating", type=int)
    render_parser.add_argument("--opponent-rating-difference", type=int)
    render_parser.add_argument("--output")
    render_parser.add_argument("--accept-casual-challenges", action="store_true")
    _add_policy_args(render_parser)

    uci_parser = subparsers.add_parser("run-uci")
    uci_parser.add_argument("--checkpoint", required=True)
    _add_policy_args(uci_parser)

    bot_parser = subparsers.add_parser("run-lichess-bot")
    bot_parser.add_argument("--checkpoint")
    bot_parser.add_argument("--max-games", type=int)
    bot_parser.add_argument("--run-hours", type=float)
    bot_parser.add_argument("--matchmaking", action="store_true")
    bot_parser.add_argument("--rated", action="store_true")
    bot_parser.add_argument("--initial-time", type=int, default=480)
    bot_parser.add_argument("--increment", type=int, default=0)
    bot_parser.add_argument("--concurrency", type=int, default=1)
    bot_parser.add_argument("--poll-interval", type=int, default=30)
    bot_parser.add_argument("--opponent-min-rating", type=int)
    bot_parser.add_argument("--opponent-max-rating", type=int)
    bot_parser.add_argument("--opponent-rating-difference", type=int)
    bot_parser.add_argument("--challenge-user")
    bot_parser.add_argument("--accept-casual-challenges", action="store_true")
    _add_policy_args(bot_parser)

    verify_parser = subparsers.add_parser("verify-bot")
    verify_parser.add_argument("--checkpoint")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    use_absolute_rating_filter = bool(
        getattr(args, "opponent_min_rating", None) is not None or getattr(args, "opponent_max_rating", None) is not None
    )
    opponent_rating_difference = getattr(args, "opponent_rating_difference", None)
    if opponent_rating_difference is None and not use_absolute_rating_filter:
        opponent_rating_difference = 500

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
                output_path=args.output,
                allow_matchmaking=args.matchmaking,
                rated=args.rated,
                initial_time=args.initial_time,
                increment=args.increment,
                concurrency=args.concurrency,
                opponent_min_rating=args.opponent_min_rating,
                opponent_max_rating=args.opponent_max_rating,
                opponent_rating_difference=opponent_rating_difference,
                accept_casual_challenges=args.accept_casual_challenges,
                selection_policy=args.selection_policy,
                min_probability=args.min_probability,
                below_threshold_weight_scale=args.below_threshold_weight_scale,
                probability_exponent=args.probability_exponent,
            )
        )
        return

    if args.command == "run-uci":
        serve_uci(
            config["__config_path__"],
            args.checkpoint,
            selection_policy=args.selection_policy,
            min_probability=args.min_probability,
            below_threshold_weight_scale=args.below_threshold_weight_scale,
            probability_exponent=args.probability_exponent,
        )
        return

    if args.command == "run-lichess-bot":
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        max_runtime_seconds = None if args.run_hours is None else max(1, int(args.run_hours * 3600))
        result = run_lichess_bot(
            config,
            checkpoint,
            max_games=args.max_games,
            max_runtime_seconds=max_runtime_seconds,
            allow_matchmaking=args.matchmaking,
            rated=args.rated,
            initial_time=args.initial_time,
            increment=args.increment,
            concurrency=args.concurrency,
            poll_interval_seconds=args.poll_interval,
            opponent_min_rating=args.opponent_min_rating,
            opponent_max_rating=args.opponent_max_rating,
            opponent_rating_difference=opponent_rating_difference,
            challenge_username=args.challenge_user,
            accept_casual_challenges=args.accept_casual_challenges,
            selection_policy=args.selection_policy,
            min_probability=args.min_probability,
            below_threshold_weight_scale=args.below_threshold_weight_scale,
            probability_exponent=args.probability_exponent,
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
