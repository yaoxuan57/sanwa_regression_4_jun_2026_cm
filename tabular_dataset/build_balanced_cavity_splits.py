"""Build balanced cavity parquets using splits propagated from trolley_regression_splits_combined."""
import argparse
from pathlib import Path

from build_regression_dataset import build_balanced_cavity_splits, load_cavity_split_configs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build balanced cavity train/val/test parquets for fine_tune.py."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="tabular_dataset directory",
    )
    parser.add_argument(
        "--cavity",
        type=int,
        choices=range(1, 9),
        nargs="+",
        default=None,
        help="Cavities to build (default: all 1-8)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cavities = args.cavity if args.cavity is not None else [int(c["cavity"]) for c in load_cavity_split_configs()]

    for cavity in cavities:
        print(f"\n{'=' * 60}\nBuilding balanced cavity {cavity}\n{'=' * 60}")
        out_dir = build_balanced_cavity_splits(cavity=cavity, data_root=args.data_root)
        print(f"Ready for fine_tune: --data_id trolley_regression_splits_combined/by_cavity/cavity_{cavity}/balanced")


if __name__ == "__main__":
    main()
