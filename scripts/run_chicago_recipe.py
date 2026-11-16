#!/usr/bin/env python
"""Run the packaged Chicago recipe; pass --input and --output to override defaults."""

import sys
from pathlib import Path

from shikong.cli.main import app


def main() -> None:
    """Execute the same validated pipeline as the shikong CLI."""
    root = Path(__file__).resolve().parent.parent
    app(
        [
            "run",
            "--dataset",
            "chicago_crime",
            "--recipe",
            "chicago_crime_v1",
            "--config",
            str(root / "configs/recipes/chicago_crime_v1.yaml"),
            "--dataset-config",
            str(root / "configs/datasets/chicago_crime_example.yaml"),
            *sys.argv[1:],
        ]
    )


if __name__ == "__main__":
    main()
