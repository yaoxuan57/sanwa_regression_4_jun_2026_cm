"""Build all cavity regression splits using seeds from split_seeds_all_cavities.json."""
import argparse
import subprocess
import sys
from pathlib import Path

from build_regression_dataset import load_cavity_split_configs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build train/val/test parquet for cavities 1-8 using combined-folder seeds."
    )
    parser.add_argument(
        "--cavity",
        type=int,
        choices=range(1, 9),
        nargs="+",
        default=None,
        help="Which cavities to build (default: all 1-8)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cavities = args.cavity if args.cavity is not None else [int(c["cavity"]) for c in load_cavity_split_configs()]
    script = Path(__file__).resolve().parent / "build_regression_dataset.py"

    for cavity in cavities:
        print(f"\n{'=' * 60}\nBuilding cavity {cavity}\n{'=' * 60}")
        cmd = [sys.executable, str(script), "--cavity", str(cavity), "--split"]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
