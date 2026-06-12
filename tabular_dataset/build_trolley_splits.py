"""Build reproducible train/val/test parquet splits for trolley mould 1 and mould 2."""
import argparse
from pathlib import Path

from build_regression_dataset import (
    TROLLEY_IMM_FEATURE_COLS,
    TROLLEY_LABEL_COLS,
    build_regression_parquet,
    split_parquet_by_indices,
    streaming_split_indices,
)


MOULD_CONFIGS = {
    "mould_1": {
        "excel": "trolley_02_aligned_with_imm_YX_mould_1.xlsx",
        "splits_dir": "trolley_regression_splits_mould_1",
    },
    "mould_2": {
        "excel": "trolley_02_aligned_with_imm_YX_mould_2.xlsx",
        "splits_dir": "trolley_regression_splits_mould_2",
    },
}


def build_one_mould(
    excel_path: Path,
    splits_dir: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    sequence_length: int,
):
    dataset_parquet = splits_dir / "full_dataset.parquet"
    build_regression_parquet(
        excel_path=excel_path,
        output_parquet=dataset_parquet,
        feature_cols=TROLLEY_IMM_FEATURE_COLS,
        label_cols=TROLLEY_LABEL_COLS,
        sequence_length=sequence_length,
    )

    import pyarrow.parquet as pq

    n_rows = pq.read_table(dataset_parquet).num_rows
    idx_train, idx_val, idx_test = streaming_split_indices(
        n_rows=n_rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    split_parquet_by_indices(
        input_path=dataset_parquet,
        out_dir=splits_dir,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        seed=seed,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build seeded train/val/test splits for trolley mould 1 and mould 2."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the Excel files",
    )
    parser.add_argument(
        "--mould",
        type=str,
        choices=["mould_1", "mould_2", "both"],
        default="both",
        help="Which mould dataset to build",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (streaming split; seed 42 matches original trolley_regression_splits)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--sequence-length", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    moulds = list(MOULD_CONFIGS) if args.mould == "both" else [args.mould]

    for mould_name in moulds:
        cfg = MOULD_CONFIGS[mould_name]
        excel_path = args.data_root / cfg["excel"]
        splits_dir = args.data_root / cfg["splits_dir"]
        if not excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")

        print(f"\n=== Building {mould_name} from {excel_path.name} ===")
        build_one_mould(
            excel_path=excel_path,
            splits_dir=splits_dir,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            sequence_length=args.sequence_length,
        )


if __name__ == "__main__":
    main()
