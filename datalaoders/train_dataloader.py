import glob

import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import pyarrow.parquet as pq
from collections import defaultdict
import re
import random


def compute_zscore_stats(x_np):
    """Per-channel mean/std over samples and time steps. x_np: [N, C, L]."""
    mean = x_np.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = x_np.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_zscore(x_np, mean, std):
    return ((x_np - mean) / std).astype(np.float32)


def compute_standard_stats(arr_np):
    """Per-column mean/std. arr_np: [N, D]."""
    mean = arr_np.mean(axis=0, keepdims=True).astype(np.float32)
    std = arr_np.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_standard(arr_np, mean, std):
    return ((arr_np - mean) / std).astype(np.float32)


class PHMDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, args, data_type, input_norm_stats=None, target_norm_stats=None):
        super(PHMDataset, self).__init__()
        self.args = args
        self.y_data_raw = None

        if args.data_percentage == "100" or data_type != "train":
            data_file = os.path.join(args.data_path, args.data_id, f"{data_type}.parquet")
            print(f"Loading full {data_type} set of {os.path.basename(os.path.dirname(data_file))} data ...")

        elif data_type == "train" and "shot" in args.data_percentage:
            data_file = os.path.join(args.data_path, args.data_id, f"{data_type}.parquet")
            print(f"Loading full {data_type} set ... now preparing the few-shot samples ...")

        else:
            data_file = os.path.join(args.data_path, args.data_id, f"{data_type}_{args.data_percentage}p.parquet")
            print(f"Loading only {args.data_percentage}% from {data_type} set of {os.path.basename(os.path.dirname(data_file))} data ...")

        # Read .parquet data
        data_file = pq.read_table(data_file)

        # Extract the samples and labels
        x_np_list = data_file['samples'].to_pylist()
        y_np = None
        if 'labels' in data_file.column_names:
            y_py_list = data_file['labels'].to_pylist()
            y_np = np.array(y_py_list)

            # Handle list-like labels stored in parquet (e.g., scalar or vector per row).
            if y_np.dtype == object:
                if args.task_type == 'FD':
                    y_np = np.array([np.array(v).reshape(-1)[0] for v in y_py_list], dtype=np.int64)
                else:
                    y_np = np.array(
                        [np.array(v, dtype=np.float32).reshape(-1) for v in y_py_list],
                        dtype=np.float32,
                    )
            else:
                if args.task_type == 'FD':
                    y_np = y_np.astype(np.int64)
                else:
                    y_np = y_np.astype(np.float32)

            if y_np.ndim > 1 and y_np.shape[-1] == 1:
                y_np = y_np.squeeze(-1)

        x_np = np.array(x_np_list, dtype=np.float32)  # [num_samples, num_channels, seq_length]

        if getattr(args, "model_arch", "transformer") == "mlp":
            x_np = x_np.mean(axis=-1).astype(np.float32)  # [num_samples, num_features]

        if getattr(args, "input_normalize", "none") == "zscore":
            if input_norm_stats is None:
                if x_np.ndim == 2:
                    self.input_norm_mean, self.input_norm_std = compute_standard_stats(x_np)
                else:
                    self.input_norm_mean, self.input_norm_std = compute_zscore_stats(x_np)
            else:
                self.input_norm_mean, self.input_norm_std = input_norm_stats
            if x_np.ndim == 2:
                x_np = apply_standard(x_np, self.input_norm_mean, self.input_norm_std)
            else:
                x_np = apply_zscore(x_np, self.input_norm_mean, self.input_norm_std)
            if data_type == "train":
                print(
                    "Applied z-score input normalization "
                    f"(per-channel, fit on {data_type}): "
                    f"mean range [{self.input_norm_mean.min():.4f}, {self.input_norm_mean.max():.4f}], "
                    f"std range [{self.input_norm_std.min():.4f}, {self.input_norm_std.max():.4f}]"
                )
            else:
                print(f"Applied z-score input normalization (stats from train set) to {data_type} set")
        elif getattr(args, "input_normalize", "none") == "standard":
            if input_norm_stats is None:
                self.input_norm_mean, self.input_norm_std = compute_standard_stats(x_np)
            else:
                self.input_norm_mean, self.input_norm_std = input_norm_stats
            x_np = apply_standard(x_np, self.input_norm_mean, self.input_norm_std)
            if data_type == "train":
                print(f"Applied StandardScaler input normalization (fit on {data_type})")
            else:
                print(f"Applied StandardScaler input normalization (stats from train set) to {data_type} set")

        if y_np is not None and getattr(args, "target_normalize", "none") == "standard":
            y_raw = y_np.astype(np.float32)
            if y_raw.ndim == 1:
                y_raw = y_raw[:, None]
            if target_norm_stats is None:
                self.target_norm_mean, self.target_norm_std = compute_standard_stats(y_raw)
            else:
                self.target_norm_mean, self.target_norm_std = target_norm_stats
            y_np = apply_standard(y_raw, self.target_norm_mean, self.target_norm_std)
            self.y_data_raw = torch.tensor(y_raw)
            if data_type == "train":
                print(f"Applied StandardScaler target normalization (fit on {data_type})")
            else:
                print(f"Applied StandardScaler target normalization (stats from train set) to {data_type} set")

        if data_type == "train" and "shot" in args.data_percentage:
            x_np, y_np = self.extract_few_shot_samples(x_np, y_np, args.data_percentage)
            print("Extracted few-shots ...")

        x_data = torch.tensor(x_np)
        y_data = torch.tensor(y_np) if y_np is not None else None

        print(f"data shapes: {x_data.shape}, {y_data.shape}")

        # Update class attributes
        x_data = x_data.to(torch.bfloat16)
        self.x_data = x_data.float()
        self.y_data = y_data if y_data is not None else None
        print("================")
        self.len = x_data.shape[0]

    def extract_few_shot_samples(self, x, y, data_percentage):
        """
        Apply few-shot sampling to the dataset.
        `args.data_percentage` is treated as the x-shot value.
        """
        match = re.search(r"(\d+)shot", data_percentage)
        shots = int(match.group(1)) if match else None
        if shots is None:
            raise ValueError(f"Invalid data_percentage format for few-shot: {self.args.data_percentage}")

        grouped = defaultdict(list)

        # Group samples by their labels
        for sample, label in zip(x, y):
            grouped[label.item()].append(sample)

        # Collect few-shot samples
        few_shot_samples = []
        few_shot_labels = []
        for label, samples in grouped.items():
            # Randomly select k-shots samples
            index = random.sample(range(0,len(samples)),shots)
            selected_samples = [samples[i] for i in index]
            # selected_samples = samples[:shots]  # Select up to `shots` samples per class
            few_shot_samples.extend(selected_samples)
            few_shot_labels.extend([label] * len(selected_samples))

        # Update dataset with few-shot samples
        x_few = np.array(few_shot_samples)
        y_few = np.array(few_shot_labels)
        return x_few, y_few

    def __getitem__(self, index):
        x = self.x_data[index]
        if x.ndim > 1 and getattr(self.args, "model_arch", "transformer") != "mlp":
            x = x.squeeze(-1)
        return x, self.y_data[index]

    def __len__(self):
        return self.len

def get_datasets(args):

    train_dataset = PHMDataset(args, data_type='train')
    input_norm_stats = None
    if getattr(args, "input_normalize", "none") in ("zscore", "standard"):
        input_norm_stats = (train_dataset.input_norm_mean, train_dataset.input_norm_std)
    target_norm_stats = None
    if getattr(args, "target_normalize", "none") == "standard":
        target_norm_stats = (train_dataset.target_norm_mean, train_dataset.target_norm_std)
    val_dataset = PHMDataset(
        args, data_type='val', input_norm_stats=input_norm_stats, target_norm_stats=target_norm_stats
    )
    test_dataset = PHMDataset(
        args, data_type='test', input_norm_stats=input_norm_stats, target_norm_stats=target_norm_stats
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print("lens:", len(train_dataset), len(val_dataset), len(test_dataset))
    print("loader lens:", len(DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False)),len(DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False)))

    return train_loader, val_loader, test_loader


def get_single_dataset(args, data_type='train'):
    dataset = PHMDataset(args, data_type=data_type)
    shuffle = True if data_type=='train' else False
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)
    return data_loader

import math
def get_class_weight(labels_dict):
    total = sum(labels_dict.values())
    max_num = max(labels_dict.values())
    mu = 1.0 / (total / max_num)
    class_weight = dict()
    for key, value in labels_dict.items():
        score = math.log(mu * total / float(value))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight
