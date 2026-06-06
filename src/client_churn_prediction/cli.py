"""Command line interface for the project workflow."""

from __future__ import annotations

import argparse

from .config import load_config
from .data import write_profile
from .models import predict, train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the project workflow.")
    parser.add_argument("--config", default=None, help="Path to configs/project.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("profile", help="Write reports/data_profile.json")
    sub.add_parser("train", help="Train the model or run the modeling routine")
    analyze_parser = sub.add_parser("analyze", help="Run AB-test or elasticity analysis")
    predict_parser = sub.add_parser("predict", help="Batch predictions")
    predict_parser.add_argument("--input", required=True, help="Input CSV")
    predict_parser.add_argument("--output", default=None, help="Output CSV")
    sub.add_parser("validate-config", help="Load config and print project metadata")

    args = parser.parse_args(argv)

    if args.command == "profile":
        print(write_profile(args.config))
    elif args.command == "train":
        print(train(args.config))
    elif args.command == "analyze":
        from .analysis import run_analysis

        print(run_analysis(args.config))
    elif args.command == "predict":
        print(predict(args.config, args.input, args.output))
    elif args.command == "validate-config":
        config = load_config(args.config)
        print(config.get("project", {}))


if __name__ == "__main__":
    main()
