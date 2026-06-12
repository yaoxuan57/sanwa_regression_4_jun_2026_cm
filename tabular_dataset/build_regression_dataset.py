import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split


TROLLEY_IMM_FEATURE_COLS = [
    "IMM_max_injection_pressure",
    "IMM_switchover_pressure",
    "IMM_end_of_packing_stroke",
    "IMM_plastification_time",
    "IMM_injection_time",
    "IMM_nozzle_temperature",
    "IMM_switchover_position",
    "IMM_barrel_front_temperature",
    "IMM_barrel_center_temperature",
    "IMM_barrel_rear_temperature",
    "IMM_feeder_temperature",
]

TROLLEY_LABEL_COLS = [
    "CSlot_Top_Gap",
    "CSlot_Length",
    "CSlot_Bot_Gap",
    "Latch_Height",
    "Latch_Length",
    "Nozzle_Diameter",
]

COMBINED_SPLITS_ROOT = Path("trolley_regression_splits_combined")
CAVITY_SPLIT_SEEDS_JSON = COMBINED_SPLITS_ROOT / "by_cavity" / "split_seeds_all_cavities.json"


def load_cavity_split_configs(seeds_json: Path = CAVITY_SPLIT_SEEDS_JSON) -> list[dict]:
    seeds_json = Path(seeds_json)
    if not seeds_json.exists():
        raise FileNotFoundError(f"Cavity split seed file not found: {seeds_json}")
    with open(seeds_json, encoding="utf-8") as f:
        return json.load(f)


def get_cavity_split_config(cavity: int, seeds_json: Path = CAVITY_SPLIT_SEEDS_JSON) -> dict:
    configs = load_cavity_split_configs(seeds_json)
    for cfg in configs:
        if int(cfg["cavity"]) == int(cavity):
            return cfg
    available = [int(c["cavity"]) for c in configs]
    raise ValueError(f"No split config for cavity {cavity}. Available: {available}")


def resolve_cavity_feature_cols(excel_path: Path) -> list[str]:
    """Use trolley IMM features; swap in IMM_cooling_time or IMM_production_cycle if needed."""
    df = pd.read_excel(excel_path, nrows=1)
    available = set(df.columns)
    feature_cols = [c for c in TROLLEY_IMM_FEATURE_COLS if c in available]
    if len(feature_cols) == len(TROLLEY_IMM_FEATURE_COLS):
        return feature_cols
    if "IMM_cooling_time" in available and "IMM_cooling_time" not in feature_cols:
        feature_cols.append("IMM_cooling_time")
    elif "IMM_production_cycle" in available and "IMM_production_cycle" not in feature_cols:
        feature_cols.append("IMM_production_cycle")
    missing = [c for c in TROLLEY_IMM_FEATURE_COLS if c not in available]
    if missing:
        print(f"Warning: Excel missing IMM columns (skipped): {missing}")
    if len(feature_cols) == 0:
        raise ValueError(f"No usable IMM feature columns found in {excel_path}")
    return feature_cols


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


def streaming_split_indices(
    n_rows: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """Per-row random assignment (same method as original trolley_regression_splits)."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) >= 1e-9:
        raise ValueError("Ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    r = rng.random(n_rows)
    t1 = train_ratio
    t2 = train_ratio + val_ratio
    idx_train = np.sort(np.where(r < t1)[0])
    idx_val = np.sort(np.where((r >= t1) & (r < t2))[0])
    idx_test = np.sort(np.where(r >= t2)[0])
    return idx_train, idx_val, idx_test


def make_reproducible_split_indices(
    n_rows: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """Nested sklearn split: first hold out test, then split train/val from the remainder."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) >= 1e-9:
        raise ValueError("Ratios must sum to 1.0")

    indices = np.arange(n_rows, dtype=np.int64)
    idx_trainval, idx_test = train_test_split(
        indices, test_size=test_ratio, random_state=seed
    )
    relative_val = val_ratio / (train_ratio + val_ratio)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=relative_val, random_state=seed
    )
    return np.sort(idx_train), np.sort(idx_val), np.sort(idx_test), np.sort(idx_trainval)


def assign_balanced_splits_from_source(
    balanced_excel: Path,
    orig_excel: Path,
    source_split_csv: Path,
    match_cols: list[str] | None = None,
):
    """Propagate train/val/test from unbalanced split onto balanced rows via label matching."""
    if match_cols is None:
        match_cols = ["product_idx"] + TROLLEY_LABEL_COLS

    orig_df = pd.read_excel(orig_excel)
    bal_df = pd.read_excel(balanced_excel)
    split_df = pd.read_csv(source_split_csv)

    missing = [c for c in match_cols if c not in orig_df.columns or c not in bal_df.columns]
    if missing:
        match_cols = [c for c in TROLLEY_LABEL_COLS if c in orig_df.columns and c in bal_df.columns]
        if not match_cols:
            raise ValueError(f"No usable match columns between {orig_excel.name} and {balanced_excel.name}")

    orig_df = orig_df.reset_index().rename(columns={"index": "orig_idx"})
    merged = bal_df.reset_index().rename(columns={"index": "balanced_idx"}).merge(
        orig_df[match_cols + ["orig_idx"]],
        on=match_cols,
        how="left",
    )
    if merged["orig_idx"].isna().any():
        n_bad = int(merged["orig_idx"].isna().sum())
        raise ValueError(f"{n_bad} balanced rows could not be matched to original rows in {balanced_excel.name}")

    orig_split = dict(zip(split_df["excel_row_index"], split_df["split"]))
    merged["split"] = merged["orig_idx"].astype(int).map(orig_split)
    if merged["split"].isna().any():
        raise ValueError("Some matched original rows are missing from source split CSV")

    idx_train = np.sort(merged.loc[merged["split"] == "train", "balanced_idx"].to_numpy(dtype=np.int64))
    idx_val = np.sort(merged.loc[merged["split"] == "val", "balanced_idx"].to_numpy(dtype=np.int64))
    idx_test = np.sort(merged.loc[merged["split"] == "test", "balanced_idx"].to_numpy(dtype=np.int64))
    return idx_train, idx_val, idx_test, merged


def build_balanced_cavity_splits(
    cavity: int,
    data_root: Path,
    sequence_length: int = 1024,
    compression: str = "zstd",
):
    cfg = get_cavity_split_config(cavity, data_root / CAVITY_SPLIT_SEEDS_JSON)
    seed = int(cfg["seed"])
    cavity_dir = data_root / COMBINED_SPLITS_ROOT / "by_cavity" / f"cavity_{cavity}"
    balanced_excel = data_root / f"cavity_{cavity}_balanced.xlsx"
    orig_excel = data_root / cfg["data_file"]
    source_split_csv = cavity_dir / f"split_indices_seed{seed}.csv"

    if not balanced_excel.exists():
        raise FileNotFoundError(f"Balanced Excel not found: {balanced_excel}")
    if not source_split_csv.exists():
        raise FileNotFoundError(f"Source split CSV not found: {source_split_csv}")

    out_dir = cavity_dir / "balanced"
    feature_cols = resolve_cavity_feature_cols(balanced_excel)
    dataset_parquet = out_dir / "full_dataset.parquet"

    build_regression_parquet(
        excel_path=balanced_excel,
        output_parquet=dataset_parquet,
        feature_cols=feature_cols,
        label_cols=TROLLEY_LABEL_COLS,
        sequence_length=sequence_length,
    )

    idx_train, idx_val, idx_test, merged = assign_balanced_splits_from_source(
        balanced_excel=balanced_excel,
        orig_excel=orig_excel,
        source_split_csv=source_split_csv,
    )

    split_parquet_by_indices(
        input_path=dataset_parquet,
        out_dir=out_dir,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        seed=seed,
        compression=compression,
        split_csv_name=f"split_indices_balanced_seed{seed}.csv",
        manifest_extra={
            "cavity": cavity,
            "data_file": balanced_excel.name,
            "split_method": "propagated_from_unbalanced",
            "source_split": source_split_csv.name,
            "source_data_file": orig_excel.name,
        },
        manifest_name=f"split_manifest_balanced_seed{seed}.json",
    )

    merged[["balanced_idx", "orig_idx", "split"]].to_csv(
        out_dir / f"balanced_to_orig_mapping_seed{seed}.csv", index=False
    )
    return out_dir


def split_parquet_by_indices(
    input_path: Path,
    out_dir: Path,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    seed: int = 42,
    compression: str = "zstd",
    split_csv_name: str | None = None,
    manifest_name: str | None = None,
    manifest_extra: dict | None = None,
):
    """Split parquet rows using fixed row indices and save reproducibility artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(input_path)
    n_rows = table.num_rows
    for name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        if idx.min() < 0 or idx.max() >= n_rows:
            raise ValueError(f"{name} indices out of range for {n_rows} rows")

    train_tbl = table.take(pa.array(idx_train, type=pa.int64()))
    val_tbl = table.take(pa.array(idx_val, type=pa.int64()))
    test_tbl = table.take(pa.array(idx_test, type=pa.int64()))

    train_path = out_dir / "train_1p.parquet"
    val_path = out_dir / "val.parquet"
    test_path = out_dir / "test.parquet"
    pq.write_table(train_tbl, train_path, compression=compression)
    pq.write_table(val_tbl, val_path, compression=compression)
    pq.write_table(test_tbl, test_path, compression=compression)

    idx_trainval = np.sort(np.concatenate([idx_train, idx_val]))
    np.save(out_dir / "train_indices.npy", idx_train)
    np.save(out_dir / "val_indices.npy", idx_val)
    np.save(out_dir / "test_indices.npy", idx_test)
    np.save(out_dir / "notebook_train_indices.npy", idx_trainval)
    np.save(out_dir / "notebook_test_indices.npy", idx_test)

    split_rows = []
    for split_name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        for row_idx in idx:
            split_rows.append(
                {"excel_row_index": int(row_idx), "parquet_row_index": int(row_idx), "split": split_name}
            )
    split_df = pd.DataFrame(split_rows).sort_values("excel_row_index").reset_index(drop=True)
    split_csv = out_dir / (split_csv_name or f"split_indices_seed{seed}.csv")
    split_df.to_csv(split_csv, index=False)

    manifest = {
        "seed": seed,
        "n_rows": int(n_rows),
        "train_rows": int(len(idx_train)),
        "val_rows": int(len(idx_val)),
        "test_rows": int(len(idx_test)),
        "notebook_train_rows": int(len(idx_trainval)),
        "notebook_test_rows": int(len(idx_test)),
        "train_ratio": float(len(idx_train) / n_rows),
        "val_ratio": float(len(idx_val) / n_rows),
        "test_ratio": float(len(idx_test) / n_rows),
        "split_csv": split_csv.name,
        "split_method": "streaming_random",
        "note": "notebook uses notebook_train_indices (=train+val) and notebook_test_indices (same test rows as fine_tune)",
    }
    if manifest_extra:
        manifest.update(manifest_extra)
        manifest["split_method"] = manifest_extra.get("split_method", manifest["split_method"])
    manifest_path = out_dir / (manifest_name or f"split_manifest_seed{seed}.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Done splitting by fixed indices.")
    print(f"train: {len(idx_train)} ({len(idx_train) / n_rows:.2%})")
    print(f"val:   {len(idx_val)} ({len(idx_val) / n_rows:.2%})")
    print(f"test:  {len(idx_test)} ({len(idx_test) / n_rows:.2%})")
    print(f"notebook train (train+val): {len(idx_trainval)} ({len(idx_trainval) / n_rows:.2%})")
    print(f"Saved splits to: {out_dir}")
    print(f"Saved index files: {split_csv.name}, {manifest_path.name}")

    return train_path, val_path, test_path, manifest_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build regression parquet dataset from Excel and optionally split into train/val/test."
    )

    parser.add_argument(
        "--cavity",
        type=int,
        choices=range(1, 9),
        default=None,
        help="Cavity 1-8: auto-load excel, features, labels, seed from trolley_regression_splits_combined",
    )
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=None,
        help="Path to input Excel file (e.g. cavity_1.xlsx). Auto-set when --cavity is used.",
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
        default=None,
        help="Feature columns from Excel (space-separated list). Auto-set for --cavity.",
    )
    parser.add_argument(
        "--label-cols",
        nargs="+",
        default=None,
        help="Label columns from Excel. Auto-set to trolley 6-dim labels for --cavity.",
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
        default=None,
        help="Output directory for split parquet files. Auto-set for --cavity.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test ratio")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for split. Auto-loaded from split_seeds_all_cavities.json for --cavity.",
    )
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size for streaming split")
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        help="Parquet compression for split outputs (e.g., zstd, snappy, gzip)",
    )

    return parser.parse_args()


def apply_cavity_defaults(args, data_root: Path):
    cfg = get_cavity_split_config(args.cavity, data_root / CAVITY_SPLIT_SEEDS_JSON)
    cavity = int(cfg["cavity"])
    if args.excel_path is None:
        args.excel_path = data_root / cfg["data_file"]
    if args.dataset_parquet == Path("regression_dataset_cavity_8.parquet"):
        args.dataset_parquet = data_root / f"regression_dataset_cavity_{cavity}.parquet"
    if args.splits_dir is None:
        args.splits_dir = data_root / COMBINED_SPLITS_ROOT / "by_cavity" / f"cavity_{cavity}"
    if args.seed is None:
        args.seed = int(cfg["seed"])
    if args.columns is None:
        args.columns = resolve_cavity_feature_cols(args.excel_path)
    if args.label_cols is None:
        args.label_cols = TROLLEY_LABEL_COLS
    if args.train_ratio == 0.6 and args.val_ratio == 0.2 and args.test_ratio == 0.2:
        args.train_ratio = float(cfg.get("train_ratio", 0.6))
        args.val_ratio = float(cfg.get("val_ratio", 0.2))
        args.test_ratio = float(cfg.get("test_ratio", 0.2))
    print(f"Using cavity {cavity} split seed {args.seed} from {CAVITY_SPLIT_SEEDS_JSON}")
    return cfg


def main():
    args = parse_args()
    data_root = Path(__file__).resolve().parent

    if args.cavity is not None:
        apply_cavity_defaults(args, data_root)
    else:
        if args.excel_path is None:
            raise ValueError("Provide --excel-path or --cavity")
        if args.columns is None or args.label_cols is None:
            raise ValueError("Provide --columns and --label-cols, or use --cavity")
        if args.splits_dir is None:
            args.splits_dir = Path("regression_splits")
        if args.seed is None:
            args.seed = 42

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
        import pyarrow.parquet as pq

        n_rows = pq.read_table(dataset_path).num_rows
        idx_train, idx_val, idx_test = streaming_split_indices(
            n_rows=n_rows,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        split_parquet_by_indices(
            input_path=dataset_path,
            out_dir=args.splits_dir,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

# Cavity 1 with seed from trolley_regression_splits_combined (seed 4201):
# python build_regression_dataset.py --cavity 1 --split
#
# All cavities 1-8:
# python build_cavity_splits.py