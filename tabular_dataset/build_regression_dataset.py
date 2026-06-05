import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def build_regression_parquet(
    excel_path: Path,
    output_parquet: Path,
    feature_cols: list[str],
    label_cols: list[str],
    sequence_length: int = 1024,
) -> Path:
    """Read excel data, expand features to (num_features, sequence_length), and save parquet."""
    if len(feature_cols) == 0:
        raise ValueError("feature columns cannot be empty")
    if len(label_cols) == 0:
        raise ValueError("label_cols cannot be empty")

    feature_set = set(feature_cols)
    label_set = set(label_cols)
    overlap = sorted(feature_set.intersection(label_set))
    if overlap:
        raise ValueError(f"A column cannot be both feature and label. Overlap: {overlap}")

    selected_cols = feature_cols + [c for c in label_cols if c not in feature_set]

    df = pd.read_excel(excel_path)

    missing_in_excel = [c for c in selected_cols if c not in df.columns]
    if missing_in_excel:
        raise ValueError(f"Columns not found in Excel: {missing_in_excel}")

    df = df[selected_cols].copy()

    samples_list = []
    labels_list = []

    for _, row in df.iterrows():
        feature_values = row[feature_cols].to_numpy(dtype=np.float32)
        sample = np.repeat(feature_values[:, None], sequence_length, axis=1).astype(np.float32)
        label = row[label_cols].to_numpy(dtype=np.float32)

        samples_list.append(sample.tolist())
        labels_list.append(label.tolist())

    parquet_df = pd.DataFrame({
        "samples": samples_list,
        "labels": labels_list,
    })

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet_df.to_parquet(output_parquet, engine="pyarrow", index=False)

    print(f"Saved dataset parquet: {output_parquet}")
    print(f"Rows: {len(parquet_df)}")
    print(f"Features: {len(feature_cols)} columns")
    print(f"Labels: {len(label_cols)} columns")
    if len(parquet_df) > 0:
        print(f"Type(samples[0]): {type(parquet_df.iloc[0]['samples'])}")
        print(f"samples[0] feature dims: {len(parquet_df.iloc[0]['samples'])} x {len(parquet_df.iloc[0]['samples'][0])}")
        print(f"labels[0] dims: {len(parquet_df.iloc[0]['labels'])}")

    return output_parquet


def split_parquet_random_streaming(
    input_path: Path,
    out_dir: Path,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    batch_size: int = 8192,
    compression: str = "zstd",
):
    """Split parquet rows into train/val/test with streaming batches."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) >= 1e-9:
        raise ValueError("Ratios must sum to 1.0")

    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_1p.parquet"
    val_path = out_dir / "val.parquet"
    test_path = out_dir / "test.parquet"

    pf = pq.ParquetFile(str(input_path))
    schema = pf.schema_arrow

    train_writer = pq.ParquetWriter(str(train_path), schema=schema, compression=compression)
    val_writer = pq.ParquetWriter(str(val_path), schema=schema, compression=compression)
    test_writer = pq.ParquetWriter(str(test_path), schema=schema, compression=compression)

    rng = np.random.default_rng(seed)
    t1 = train_ratio
    t2 = train_ratio + val_ratio

    n_train = 0
    n_val = 0
    n_test = 0

    try:
        for rb in pf.iter_batches(batch_size=batch_size):
            n = rb.num_rows
            r = rng.random(n)

            m_train = pa.array(r < t1)
            m_val = pa.array((r >= t1) & (r < t2))
            m_test = pa.array(r >= t2)

            tbl = pa.Table.from_batches([rb])

            if pc.any(m_train).as_py():
                train_tbl = pc.filter(tbl, m_train)
                train_writer.write_table(train_tbl)
                n_train += train_tbl.num_rows

            if pc.any(m_val).as_py():
                val_tbl = pc.filter(tbl, m_val)
                val_writer.write_table(val_tbl)
                n_val += val_tbl.num_rows

            if pc.any(m_test).as_py():
                test_tbl = pc.filter(tbl, m_test)
                test_writer.write_table(test_tbl)
                n_test += test_tbl.num_rows
    finally:
        train_writer.close()
        val_writer.close()
        test_writer.close()

    total = n_train + n_val + n_test
    print("Done splitting.")
    print(f"train: {n_train} ({n_train / total:.2%})")
    print(f"val:   {n_val} ({n_val / total:.2%})")
    print(f"test:  {n_test} ({n_test / total:.2%})")
    print(f"Saved splits to: {out_dir}")

    return train_path, val_path, test_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build regression parquet dataset from Excel and optionally split into train/val/test."
    )

    parser.add_argument(
        "--excel-path",
        type=Path,
        required=True,
        help="Path to input Excel file (e.g. cavity_8.xlsx)",
    )
    parser.add_argument(
        "--dataset-parquet",
        type=Path,
        default=Path("regression_dataset_cavity_8.parquet"),
        help="Output parquet path for generated dataset",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        required=True,
        help="Feature columns from Excel (space-separated list)",
    )
    parser.add_argument(
        "--label-cols",
        nargs="+",
        required=True,
        help="Label columns from Excel. Any number is allowed.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1024,
        help="Length to repeat each feature value across time axis",
    )

    parser.add_argument(
        "--split",
        action="store_true",
        help="If set, also split generated dataset parquet into train/val/test files",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("regression_splits"),
        help="Output directory for split parquet files",
    )
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test ratio")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for split")
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size for streaming split")
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        help="Parquet compression for split outputs (e.g., zstd, snappy, gzip)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {args.excel_path}")

    dataset_path = build_regression_parquet(
        excel_path=args.excel_path,
        output_parquet=args.dataset_parquet,
        feature_cols=args.columns,
        label_cols=args.label_cols,
        sequence_length=args.sequence_length,
    )

    if args.split:
        split_parquet_random_streaming(
            input_path=dataset_path,
            out_dir=args.splits_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            batch_size=args.batch_size,
            compression=args.compression,
        )


if __name__ == "__main__":
    main()

# python build_regression_dataset.py --excel-path trolley_02_aligned_with_imm_YX_mould_1.xlsx --columns P3 P4 P9 P11 --label-cols Thickness --dataset-parquet regression_dataset_S3_Hands_on_wafer_thickness_train.parquet --split --splits-dir regression_splits_S3 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1