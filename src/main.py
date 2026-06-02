"""Main entry point for the railway crowd analytics CLI placeholder."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the placeholder runner."""
    parser = argparse.ArgumentParser(description="Railway crowd analytics placeholder app")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/app.yaml"),
        help="Path to the app YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the placeholder CLI startup output."""
    args = parse_args()
    print(f"[railway-crowd-analytics] startup placeholder using config: {args.config}")


if __name__ == "__main__":
    main()
