import os
import argparse
import datetime
import math
import csv
from pathlib import Path

import torch
import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Qt5Agg')
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from torchmetrics import MetricCollection
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from torchmetrics.classification import Accuracy, MulticlassF1Score, MulticlassConfusionMatrix
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from datalaoders.train_dataloader import get_datasets
from model.model import Transformer_bkbone
from model.mlp_regressor import MLPRegressor
from utils import save_copy_of_files, str2bool, get_rul_report, scoring_function_v2

# ==================== Model Wrapper ====================
class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if getattr(args, "model_arch", "transformer") == "mlp":
            n_inputs = args.num_channels
            n_outputs = args.num_classes
            self.model = MLPRegressor(n_inputs, n_outputs, dropout=args.mlp_dropout)
        else:
            self.model = Transformer_bkbone(args)
        if args.task_type == 'FD':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif args.task_type == 'RUL':
            if getattr(args, "regression_loss", "mse") == "mae":
                self.loss_fn = torch.nn.L1Loss()
            else:
                self.loss_fn = torch.nn.MSELoss()
        if args.task_type == 'FD':
            self.train_metrics = MetricCollection({
                "acc": Accuracy(task="multiclass", num_classes=args.num_classes),
                "f1": MulticlassF1Score(num_classes=args.num_classes, average="macro")
            })
            self.val_metrics = MetricCollection({
                "acc": Accuracy(task="multiclass", num_classes=args.num_classes),
                "f1": MulticlassF1Score(num_classes=args.num_classes, average="macro")
            })
            self.test_f1 = MulticlassF1Score(num_classes=args.num_classes, average="macro")
            self.confusion_matrix = MulticlassConfusionMatrix(num_classes=args.num_classes)
        elif args.task_type == 'RUL':
            if getattr(args, "regression_loss", "mse") == "mae":
                metric = MeanAbsoluteError()
                self.regression_metric_name = "mae"
            else:
                metric = MeanSquaredError(squared=False)
                self.regression_metric_name = "rmse"
            self.train_metrics = MetricCollection({self.regression_metric_name: metric})
            self.val_metrics = MetricCollection({self.regression_metric_name: metric})
            self.test_metric = metric



        self.total_steps = args.num_epochs * args.tl_length
        self.num_warmup_steps = int(0.1 * self.total_steps)  # 2048

        self.test_preds = []
        self.test_targets = []
        self.rul_label_shape = None
        self.test_preds_rows = []
        self.test_targets_rows = []
        self.val_preds_epoch = []
        self.val_targets_epoch = []
        self.val_rmse_per_var_history = []
        self.test_raw_targets_rows = []

    def forward(self, x):
        if getattr(self.args, "model_arch", "transformer") == "mlp":
            return self.model(x)
        return self.model(x)

    def _predict_rul(self, x):
        if getattr(self.args, "model_arch", "transformer") == "mlp":
            return self.model(x)
        feats = self(x)
        return self.model.predict(feats)

    def _denormalize_predictions(self, arr):
        if getattr(self.args, "target_normalize", "none") != "standard":
            return arr
        arr = np.asarray(arr, dtype=np.float32)
        mean = np.asarray(self.args.target_norm_mean, dtype=np.float32).reshape(-1)
        std = np.asarray(self.args.target_norm_std, dtype=np.float32).reshape(-1)
        if arr.ndim == 1:
            arr = arr.reshape(-1, mean.shape[0])
        return arr * std + mean

    def configure_optimizers(self):
        if getattr(self.args, "model_arch", "transformer") == "mlp":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.wt_decay)
            return optimizer

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.wt_decay)
        scheduler = {
            'scheduler': self.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.num_warmup_steps,
                                                              num_training_steps=self.total_steps),
            'name': 'learning_rate', 'interval': 'step', 'frequency': 1,
        }
        return [optimizer], [scheduler]


    def get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _shared_step(self, batch, stage):
        x, y = batch

        if self.args.task_type == "FD":
            # ---- target: force (B,) long ----
            if y.ndim > 1:
                # one-hot (B,K) OR (B,1)
                if y.size(-1) == 1:
                    y = y.view(-1)
                else:
                    y = torch.argmax(y, dim=1)
            y = y.long()

            # ---- forward gives features/tokens ----
            feats = self(x)

            # ---- predict() must output class logits (B,num_classes) ----
            class_logits = self.model.predict(feats)

            # fix the B=1 squeeze case: (num_classes,) -> (1,num_classes)
            if class_logits.ndim == 1:
                class_logits = class_logits.unsqueeze(0)

            # if predict returns (B,1,num_classes) etc, flatten to (B,num_classes)
            if class_logits.ndim > 2:
                class_logits = class_logits.view(class_logits.size(0), -1)

            loss = self.loss_fn(class_logits, y)

            # for metrics use class indices (B,)
            preds = torch.argmax(class_logits, dim=1)

        elif self.args.task_type == "RUL":
            preds = self._predict_rul(x).float()
            y = y.float()

            # Normalize to [B, D] so scalar and multi-target regression share one path.
            if preds.ndim == 1:
                batch_size = y.shape[0] if y.ndim > 1 else 1
                if batch_size == 1 and preds.numel() > 1:
                    preds = preds.unsqueeze(0)
                else:
                    preds = preds.unsqueeze(-1)
            else:
                preds = preds.view(preds.size(0), -1)
            y = self._prepare_rul_targets(y)

            if preds.size(1) != y.size(1):
                raise ValueError(
                    f"Prediction/target shape mismatch for RUL: preds={tuple(preds.shape)}, targets={tuple(y.shape)}"
                )

            loss = self.loss_fn(preds, y)

        # ---- metrics/logging ----
        if stage == "train":
            self.train_metrics.update(preds, y)
            self.log_dict({f"train_{k}": v for k, v in self.train_metrics.compute().items()},
                        on_epoch=True, prog_bar=True)
            self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        elif stage == "val":
            self.val_metrics.update(preds, y)
            self.log_dict({f"val_{k}": v for k, v in self.val_metrics.compute().items()},
                        on_epoch=True, prog_bar=True)
            self.log("val_loss", loss, on_epoch=True, prog_bar=True)
            if self.args.task_type == "RUL":
                self.val_preds_epoch.append(preds.detach().cpu().numpy())
                self.val_targets_epoch.append(y.detach().cpu().numpy())

        elif stage == "test":
            if self.args.task_type == "FD":
                self.test_f1.update(preds, y)
                self.confusion_matrix.update(preds, y)
                self.test_preds.extend(preds.cpu().numpy())
                self.test_targets.extend(y.cpu().numpy())

                acc = Accuracy(task="multiclass", num_classes=self.args.num_classes).to(preds.device)(preds, y)
                self.log("test_accuracy", acc)

            else:
                preds_np = preds.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
                preds_denorm = self._denormalize_predictions(preds_np)
                y_denorm = self._denormalize_predictions(y_np)
                self.test_metric.update(
                    torch.tensor(preds_denorm, device=preds.device),
                    torch.tensor(y_denorm, device=y.device),
                )
                self.test_preds.extend(preds_denorm.reshape(-1))
                self.test_targets.extend(y_denorm.reshape(-1))
                self.test_preds_rows.append(preds_denorm)
                self.test_targets_rows.append(y_denorm)
                if self.args.rul_target_mode_effective == "single":
                    score = scoring_function_v2(np.array(self.test_preds), np.array(self.test_targets))
                    self.log("test_score", score)

            self.log("test_loss", loss)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        if self.args.task_type == "RUL":
            if len(self.val_preds_epoch) > 0:
                preds_2d = np.concatenate(self.val_preds_epoch, axis=0)
                targets_2d = np.concatenate(self.val_targets_epoch, axis=0)
                per_var_rmse = self._compute_per_variable_rmse(preds_2d, targets_2d)
                self.val_rmse_per_var_history.append(per_var_rmse)
                for i, rmse_val in enumerate(per_var_rmse):
                    self.log(f"val_rmse_var{i}", float(rmse_val), on_epoch=True, prog_bar=False)

            self.val_preds_epoch = []
            self.val_targets_epoch = []
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        if self.args.task_type == 'FD':
            f1_score = self.test_f1.compute()
            self.log("test_f1", f1_score)
            self.test_f1.reset()

            fig, ax = self.confusion_matrix.plot()
            fig.tight_layout()
            fig.savefig(f"{self.args.ckpt_dir}/confusion_matrix.png", bbox_inches="tight")
            print("Test Confusion Matrix saved.")
            self.confusion_matrix.reset()

            labels = list(range(self.args.num_classes))  # [0,1,2,3]
            print("unique y_true:", np.unique(self.test_targets, return_counts=True))
            print("unique y_pred:", np.unique(self.test_preds, return_counts=True))
            print("args.num_classes:", self.args.num_classes)
            print("class_names:", self.args.class_names)

            report = classification_report(
                self.test_targets,
                self.test_preds,
                labels=labels,
                target_names=self.args.class_names,
                digits=4,
                zero_division=0
            )
            print("=== Classification Report ===")
            print(report)

            with open(f"{self.args.ckpt_dir}/classification_report.txt", "w") as f:
                f.write(report)

        elif self.args.task_type == 'RUL':
            test_metric_val = self.test_metric.compute()
            metric_name = self.regression_metric_name
            self.log(f"test_{metric_name}", test_metric_val)
            self.test_metric.reset()
            rmse = test_metric_val if metric_name == "rmse" else None

            test_preds_arr = np.array(self.test_preds, dtype=np.float32)
            test_targets_arr = np.array(self.test_targets, dtype=np.float32)

            if len(self.test_preds_rows) > 0:
                preds_2d = np.concatenate(self.test_preds_rows, axis=0)
                targets_2d = np.concatenate(self.test_targets_rows, axis=0)
            else:
                preds_2d = test_preds_arr.reshape(-1, 1)
                targets_2d = test_targets_arr.reshape(-1, 1)

            metrics_table, metrics_rows = self._build_rul_metrics_table(preds_2d, targets_2d)
            overall_metrics = next(row for row in metrics_rows if row["output"] == "OVERALL (macro avg)")
            mae = overall_metrics["MAE"]

            # Scoring function is only defined for scalar RUL.
            score = None
            if self.args.rul_target_mode_effective == "single":
                score = scoring_function_v2(test_preds_arr, test_targets_arr)
                self.log("test_score", score)
            self.log("test_mae", mae)
            self.log("test_rmse", overall_metrics["RMSE"])
            self.log("test_mse", overall_metrics["MSE"])

            # Save flattened predictions and targets.
            np.save(f"{self.args.ckpt_dir}/test_preds.npy", test_preds_arr)
            np.save(f"{self.args.ckpt_dir}/test_targets.npy", test_targets_arr)
            np.save(f"{self.args.ckpt_dir}/test_preds_2d.npy", preds_2d)
            np.save(f"{self.args.ckpt_dir}/test_targets_2d.npy", targets_2d)
            self._save_rul_metrics_csv(metrics_rows)
            if self.args.rul_target_mode_effective == "multi" and preds_2d.ndim == 2 and preds_2d.shape[1] > 1:
                self._save_multi_target_scatter_plot(preds_2d, targets_2d, float(test_metric_val))
            self._save_selected_test_window_visuals(preds_2d, targets_2d)
            self._save_val_rmse_over_epochs_plot()

            report = f"""=== RUL Prediction Report ===
Target Mode: {self.args.rul_target_mode_effective}
Output Dim: {self.args.num_classes}
Model Arch: {getattr(self.args, 'model_arch', 'transformer')}
Regression Head: {getattr(self.args, 'regression_head', 'linear')}
Regression Loss: {getattr(self.args, 'regression_loss', 'mse')}
Training Metric ({metric_name.upper()}): {float(test_metric_val):.8f}
"""
            if score is not None:
                report += f"RUL Score: {score:.8f}\n"
            else:
                report += "RUL Score: N/A (multi-target mode)\n"

            report += f"""
=== Per-target and Overall Metrics (test set) ===
{metrics_table}

Saved metrics CSV: regression_metrics.csv

=== Prediction Statistics (flattened test values) ===
True  -> min: {np.min(test_targets_arr):.4f}, max: {np.max(test_targets_arr):.4f}, mean: {np.mean(test_targets_arr):.4f}
Pred  -> min: {np.min(test_preds_arr):.4f}, max: {np.max(test_preds_arr):.4f}, mean: {np.mean(test_preds_arr):.4f}

First 10 flattened predictions (True, Predicted):
"""
            for i in range(min(10, len(test_targets_arr))):
                report += f"{test_targets_arr[i]:.4f}, {test_preds_arr[i]:.4f}\n"

            print(report)
            with open(f"{self.args.ckpt_dir}/rul_report.txt", "w") as f:
                f.write(report)
            # Single-target only: one combined scatter. Multi-target uses per-output grid plot.
            if not (self.args.rul_target_mode_effective == "multi" and len(self.test_preds_rows) > 0):
                plt.figure(figsize=(10, 6))
                plt.scatter(test_targets_arr, test_preds_arr, alpha=0.5)
                plt.plot([min(test_targets_arr), max(test_targets_arr)],
                         [min(test_targets_arr), max(test_targets_arr)], 'r--')
                plt.xlabel('True RUL')
                plt.ylabel('Predicted RUL')
                title = f'RUL Prediction\nRMSE: {overall_metrics["RMSE"]:.2f}, MAE: {mae:.2f}'
                if score is not None:
                    title += f', Score: {score:.2f}'
                plt.title(title)
                plt.tight_layout()
                plt.savefig(f"{self.args.ckpt_dir}/rul_prediction.png", bbox_inches="tight")
                plt.close()

        self.test_preds = []
        self.test_targets = []
        self.test_preds_rows = []
        self.test_targets_rows = []

    def _prepare_rul_targets(self, y):
        y = y.float()
        if y.ndim == 1:
            y2 = y.unsqueeze(-1)
            self.rul_label_shape = (1,)
            return y2

        y2 = y.view(y.size(0), -1)
        self.rul_label_shape = tuple(y.shape[1:])

        if self.args.rul_target_mode_effective == "single":
            idx = self.args.rul_single_target_index
            if idx < 0 or idx >= y2.size(1):
                raise ValueError(
                    f"rul_single_target_index={idx} out of range for labels flattened dim {y2.size(1)}"
                )
            return y2[:, idx:idx + 1]
        return y2

    def _compute_per_variable_rmse(self, preds_2d, targets_2d):
        if self.rul_label_shape is None or len(self.rul_label_shape) == 0:
            return np.array([float(np.sqrt(np.mean((preds_2d - targets_2d) ** 2)))], dtype=np.float32)

        preds_nd = preds_2d.reshape(-1, *self.rul_label_shape)
        targets_nd = targets_2d.reshape(-1, *self.rul_label_shape)

        if len(self.rul_label_shape) >= 2:
            # First axis in label shape is treated as variable/channel axis.
            rmse_by_var = []
            for var_idx in range(self.rul_label_shape[0]):
                p = preds_nd[:, var_idx, ...]
                t = targets_nd[:, var_idx, ...]
                rmse_by_var.append(float(np.sqrt(np.mean((p - t) ** 2))))
            return np.array(rmse_by_var, dtype=np.float32)

        rmse = np.sqrt(np.mean((preds_nd - targets_nd) ** 2, axis=0))
        return rmse.reshape(-1).astype(np.float32)

    def _save_selected_test_window_visuals(self, preds_2d, targets_2d):
        if self.rul_label_shape is None and self.args.rul_target_mode_effective != "single":
            return

        # Tabular / single-step regression has no forecast horizon — skip time-step plots.
        if self.rul_label_shape is not None and len(self.rul_label_shape) == 1:
            return

        if self.args.rul_target_mode_effective == "single":
            preds_nd = preds_2d.reshape(-1, 1, 1)
            targets_nd = targets_2d.reshape(-1, 1, 1)
        else:
            preds_nd = preds_2d.reshape(-1, *self.rul_label_shape)
            targets_nd = targets_2d.reshape(-1, *self.rul_label_shape)

        total_windows = preds_nd.shape[0]
        if total_windows == 0:
            return

        # Sample 20 evenly spaced windows from first 100 windows (or less if unavailable).
        sample_pool = min(100, total_windows)
        n_select = min(20, sample_pool)
        selected_idx = np.linspace(0, sample_pool - 1, num=n_select, dtype=int)

        with open(f"{self.args.ckpt_dir}/selected_test_windows_indices.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["window_index"])
            for idx in selected_idx:
                writer.writerow([int(idx)])

        if self.args.rul_target_mode_effective == "single":
            num_vars = 1
            horizon = 1
            preds_view = preds_nd.reshape(total_windows, num_vars, horizon)
            targets_view = targets_nd.reshape(total_windows, num_vars, horizon)
        elif len(self.rul_label_shape) >= 2:
            num_vars = self.rul_label_shape[0]
            horizon = int(np.prod(self.rul_label_shape[1:]))
            preds_view = preds_nd.reshape(total_windows, num_vars, horizon)
            targets_view = targets_nd.reshape(total_windows, num_vars, horizon)
        else:
            num_vars = self.rul_label_shape[0]
            horizon = 1
            preds_view = preds_nd.reshape(total_windows, num_vars, horizon)
            targets_view = targets_nd.reshape(total_windows, num_vars, horizon)

        for var_idx in range(num_vars):
            preds_sel = preds_view[selected_idx, var_idx, :]
            targets_sel = targets_view[selected_idx, var_idx, :]

            csv_path = f"{self.args.ckpt_dir}/test_selected_windows_var{var_idx}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["window_index"]
                for h in range(horizon):
                    header += [f"actual_h{h}", f"pred_h{h}"]
                writer.writerow(header)

                for i, win_idx in enumerate(selected_idx):
                    row = [int(win_idx)]
                    for h in range(horizon):
                        row += [float(targets_sel[i, h]), float(preds_sel[i, h])]
                    writer.writerow(row)

            # Save 20 PNGs per variable: one plot per selected window.
            var_dir = Path(self.args.ckpt_dir) / f"test_selected_windows_var{var_idx}"
            var_dir.mkdir(parents=True, exist_ok=True)
            horizon_x = np.arange(horizon)

            for i, win_idx in enumerate(selected_idx):
                plt.figure(figsize=(8, 4))
                plt.plot(horizon_x, targets_sel[i], marker='o', label='Actual')
                plt.plot(horizon_x, preds_sel[i], marker='x', label='Predicted')
                plt.xlabel("Horizon step")
                plt.ylabel("Value")
                plt.title(f"Variable {var_idx} - Window {int(win_idx)}")
                plt.grid(alpha=0.25)
                plt.legend(loc="best")
                plt.tight_layout()
                plt.savefig(var_dir / f"window_{int(win_idx)}.png", bbox_inches="tight")
                plt.close()

    def _compute_regression_metrics(self, actual, predicted):
        actual = np.asarray(actual, dtype=np.float64).reshape(-1)
        predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
        mae = float(mean_absolute_error(actual, predicted))
        mse = float(mean_squared_error(actual, predicted))
        rmse = float(np.sqrt(mse))
        denom = np.maximum(np.abs(actual), 1e-8)
        mape = float(np.mean(np.abs(actual - predicted) / denom) * 100.0)
        r2 = float(r2_score(actual, predicted))
        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}

    def _build_rul_metrics_table(self, preds_2d, targets_2d):
        preds_2d = np.asarray(preds_2d, dtype=np.float64)
        targets_2d = np.asarray(targets_2d, dtype=np.float64)
        if preds_2d.ndim == 1:
            preds_2d = preds_2d.reshape(-1, 1)
            targets_2d = targets_2d.reshape(-1, 1)

        num_targets = preds_2d.shape[1]
        target_names = self._get_rul_target_names(num_targets)
        metric_keys = ["MAE", "MSE", "RMSE", "MAPE", "R2"]

        rows = []
        for idx in range(num_targets):
            row = {"output": target_names[idx]}
            row.update(self._compute_regression_metrics(targets_2d[:, idx], preds_2d[:, idx]))
            rows.append(row)

        per_target_rows = rows.copy()
        overall_macro = {"output": "OVERALL (macro avg)"}
        for key in metric_keys:
            overall_macro[key] = float(np.mean([row[key] for row in per_target_rows]))
        rows.append(overall_macro)

        overall_flat = {"output": "OVERALL (all samples)"}
        overall_flat.update(self._compute_regression_metrics(targets_2d, preds_2d))
        # Pooled R2 is dominated by large-scale targets; report macro-averaged R2 instead.
        overall_flat["R2"] = overall_macro["R2"]
        rows.append(overall_flat)

        header = ["output"] + metric_keys
        col_widths = {key: len(key) for key in header}
        col_widths["output"] = max(len("output"), max(len(row["output"]) for row in rows))
        for key in metric_keys:
            col_widths[key] = max(len(key), max(len(f"{row[key]:.6f}") for row in rows))

        def fmt_header():
            cells = ["output".ljust(col_widths["output"])]
            for key in metric_keys:
                cells.append(key.rjust(col_widths[key]))
            return "  ".join(cells)

        def fmt_row(row):
            cells = [row["output"].ljust(col_widths["output"])]
            for key in metric_keys:
                cells.append(f"{row[key]:>{col_widths[key]}.6f}")
            return "  ".join(cells)

        lines = [
            fmt_header(),
            "  ".join("-" * col_widths[key] for key in header),
        ]
        lines.extend(fmt_row(row) for row in rows)
        return "\n".join(lines), rows

    def _save_rul_metrics_csv(self, rows):
        metric_keys = ["output", "MAE", "MSE", "RMSE", "MAPE", "R2"]
        csv_path = f"{self.args.ckpt_dir}/regression_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metric_keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in metric_keys})

    def _get_rul_target_names(self, num_targets):
        names = getattr(self.args, "rul_target_names", None)
        if names is None and "trolley_regression" in getattr(self.args, "data_id", ""):
            names = TROLLEY_TARGET_NAMES
        if names is not None:
            if len(names) != num_targets:
                print(
                    f"Warning: rul_target_names has {len(names)} entries but model has "
                    f"{num_targets} targets. Falling back to generic names."
                )
                return [f"target{i}" for i in range(num_targets)]
            return names
        return [f"target{i}" for i in range(num_targets)]

    def _save_multi_target_scatter_plot(self, preds_2d, targets_2d, rmse):
        num_targets = preds_2d.shape[1]
        target_names = self._get_rul_target_names(num_targets)
        if num_targets == 6:
            n_rows, n_cols, figsize = 2, 3, (14, 9)
        else:
            n_cols = 3
            n_rows = int(np.ceil(num_targets / n_cols))
            figsize = (5 * n_cols, 4 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()

        for idx in range(num_targets):
            ax = axes[idx]
            actual = targets_2d[:, idx]
            predicted = preds_2d[:, idx]

            lo = min(actual.min(), predicted.min())
            hi = max(actual.max(), predicted.max())
            pad = 0.05 * (hi - lo) if hi > lo else 0.01
            lim = (lo - pad, hi + pad)

            ax.scatter(actual, predicted, alpha=0.7, edgecolors="k", linewidths=0.3, s=40)
            ax.plot(lim, lim, "r--", lw=1.5, label="y = x")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(target_names[idx])
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)

        for idx in range(num_targets, len(axes)):
            axes[idx].axis("off")

        fig.suptitle("Actual vs Predicted (test set)", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(f"{self.args.ckpt_dir}/actual_vs_predicted_test.png", bbox_inches="tight")
        plt.close(fig)

    def _save_val_rmse_over_epochs_plot(self):
        if len(self.val_rmse_per_var_history) == 0:
            return

        rmse_hist = np.array(self.val_rmse_per_var_history, dtype=np.float32)  # [epochs, vars]
        if rmse_hist.ndim == 1:
            rmse_hist = rmse_hist[:, None]

        epochs = np.arange(1, rmse_hist.shape[0] + 1)
        plt.figure(figsize=(10, 6))
        for var_idx in range(rmse_hist.shape[1]):
            label = "target0" if (rmse_hist.shape[1] == 1) else f"var{var_idx}"
            plt.plot(epochs, rmse_hist[:, var_idx], marker='o', label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Validation RMSE")
        plt.title("Validation RMSE vs Epoch per Target Variable")
        plt.grid(alpha=0.2)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(f"{self.args.ckpt_dir}/val_rmse_per_variable_over_epochs.png", bbox_inches="tight")
        plt.close()


# ==================== Callbacks ====================
def construct_experiment_dir(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_description = "FT" if args.load_from_pretrained else "Supervised"
    run_description += f"_{getattr(args, 'model_arch', 'transformer')}_{args.model_type}"
    if getattr(args, "model_arch", "transformer") == "transformer" and getattr(args, "regression_head", "linear") == "mlp":
        run_description += "_mlphead"
    run_description += f"_{args.data_id}_from{args.pretraining_epoch_id}_{args.model_id}"
    run_description += f"_bs{args.batch_size}_lr{args.lr}_seed{args.random_seed}_{timestamp}"
    return run_description


def plot_metrics(metrics, ckpt_dir, task_type):
    plt.figure()
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.tight_layout()
    plt.savefig(f"{ckpt_dir}/loss.png", bbox_inches="tight")

    plt.figure()
    if task_type == 'FD':
        plt.plot(metrics["train_acc"], label="Train Acc")
        plt.plot(metrics["val_acc"], label="Val Acc")
        plt.legend()
        plt.title("Accuracy Curve")
    elif task_type == 'RUL':
        if "train_mae" in metrics and len(metrics["train_mae"]) > 0:
            plt.plot(metrics["train_mae"], label="Train MAE")
            plt.plot(metrics["val_mae"], label="Val MAE")
            plt.title("MAE Curve")
        else:
            plt.plot(metrics["train_rmse"], label="Train RMSE")
            plt.plot(metrics["val_rmse"], label="Val RMSE")
            plt.title("RMSE Curve")
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{ckpt_dir}/performance_metric.png", bbox_inches="tight")


class InMemoryBestWeightsCallback(pl.Callback):
    """Track best validation weights in RAM when disk checkpoints are disabled."""

    def __init__(self, monitor, mode="min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_state_dict = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.monitor not in trainer.callback_metrics:
            return
        current = trainer.callback_metrics[self.monitor].item()
        improved = current < self.best_score if self.mode == "min" else current > self.best_score
        if improved:
            self.best_score = current
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()}

    def on_train_end(self, trainer, pl_module):
        if self.best_state_dict is None:
            return
        device = next(pl_module.parameters()).device
        pl_module.load_state_dict(
            {k: v.to(device) for k, v in self.best_state_dict.items()}
        )
        print(f"Restored in-memory best weights ({self.monitor}={self.best_score:.6f})")


class MetricTrackerCallback(pl.Callback):
    def __init__(self, task_type):
        super().__init__()
        self.task_type = task_type
        self.losses = {"train_loss": [], "val_loss": []}
        if task_type == 'FD':
            self.accuracies = {"train_acc": [], "val_acc": []}
        elif task_type == 'RUL':
            self.regression_metric_name = "rmse"
            self.rmses = {"train_rmse": [], "val_rmse": [], "train_mae": [], "val_mae": []}

    def on_validation_epoch_end(self, trainer, pl_module):
        self.losses["val_loss"].append(trainer.callback_metrics["val_loss"].item())
        if self.task_type == 'FD':
            self.accuracies["val_acc"].append(trainer.callback_metrics["val_acc"].item())
        elif self.task_type == 'RUL':
            metric_name = getattr(pl_module, "regression_metric_name", "rmse")
            key = f"val_{metric_name}"
            if key in trainer.callback_metrics:
                self.rmses[key].append(trainer.callback_metrics[key].item())

    def on_train_epoch_end(self, trainer, pl_module):
        self.losses["train_loss"].append(trainer.callback_metrics["train_loss"].item())
        if self.task_type == 'FD':
            self.accuracies["train_acc"].append(trainer.callback_metrics["train_acc"].item())
        elif self.task_type == 'RUL':
            metric_name = getattr(pl_module, "regression_metric_name", "rmse")
            key = f"train_{metric_name}"
            if key in trainer.callback_metrics:
                self.rmses[key].append(trainer.callback_metrics[key].item())

# ==================== Main ====================
def main(args):
    pl.seed_everything(args.random_seed)
    train_loader, val_loader, test_loader = get_datasets(args)

    # args extracted from the running dataset
    if args.task_type == 'FD':
        args.num_classes = len(np.unique(train_loader.dataset.y_data))
        args.class_names = [str(i) for i in range(args.num_classes)]
    else:
        y_shape = tuple(train_loader.dataset.y_data.shape[1:])
        if len(y_shape) == 0:
            total_target_dim = 1
        else:
            total_target_dim = int(np.prod(y_shape))

        if args.rul_target_mode == "auto":
            args.rul_target_mode_effective = "single" if total_target_dim == 1 else "multi"
        else:
            args.rul_target_mode_effective = args.rul_target_mode

        args.num_classes = 1 if args.rul_target_mode_effective == "single" else total_target_dim
        print(f"RUL labels shape per sample: {y_shape}, flattened dim={total_target_dim}")
        print(f"RUL target mode: {args.rul_target_mode_effective}")
        if args.rul_target_mode_effective == "single":
            print(f"RUL single target index (flattened): {args.rul_single_target_index}")
    if getattr(args, "model_arch", "transformer") == "mlp":
        args.num_channels = train_loader.dataset.x_data.shape[1]
        args.seq_len = 1
    else:
        args.seq_len = train_loader.dataset.x_data.shape[-1]
        args.num_channels = train_loader.dataset.x_data.shape[1]
    args.tl_length = len(train_loader)
    if getattr(args, "target_normalize", "none") == "standard" and hasattr(train_loader.dataset, "target_norm_mean"):
        args.target_norm_mean = train_loader.dataset.target_norm_mean
        args.target_norm_std = train_loader.dataset.target_norm_std

    # Callbacks
    run_description = construct_experiment_dir(args)
    print(f"========== {run_description} ===========")
    ckpt_dir = f"checkpoints/{run_description}"

    args.ckpt_dir = ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # Set monitoring metric based on task type
    best_weights = None
    if args.task_type == 'FD':
        checkpoint = ModelCheckpoint(monitor="train_f1_epoch", mode="max", save_top_k=1, dirpath=ckpt_dir,
                                     filename="best")
        early_stop = EarlyStopping(monitor="train_f1_epoch", patience=args.patience, mode="max")
    elif args.task_type == 'RUL':
        # Skip mid-epoch checkpoint writes; they often fail with PermissionError on Desktop/OneDrive.
        checkpoint = None
        monitor = f"val_{args.regression_loss}" if args.regression_loss == "mae" else "val_rmse"
        early_stop = EarlyStopping(monitor=monitor, patience=args.patience, mode="min")
        best_weights = InMemoryBestWeightsCallback(monitor=monitor, mode="min")

    tracker = MetricTrackerCallback(args.task_type)

    if checkpoint is not None:
        save_copy_of_files(checkpoint)

    model = Model(args)

    load_map = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Optional load pretrained weights
    if getattr(args, "model_arch", "transformer") == "mlp":
        args.load_from_pretrained = False
    if args.load_from_pretrained and args.pretrained_model_type != 'mae':
        path = os.path.join(args.pretrained_model_dir, f"pretrain-epoch={args.pretraining_epoch_id}.ckpt")
        checkpoint_data = torch.load(path, map_location=load_map, weights_only=False)


        # Filter and count matching keys with the same shape
        matched_weights = {
            k: v for k, v in checkpoint_data['state_dict'].items()
            if k in model.state_dict() and model.state_dict()[k].size() == v.size()
        }

        total_pretrained = len(checkpoint_data['state_dict'])
        model.load_state_dict(matched_weights, strict=False)

        print(f"Loaded pretrained weights from {path}")
        print(f"Matched weights: {len(matched_weights)}/{len(model.state_dict())} model parameters matched "
              f"(from {total_pretrained} pretrained parameters)")
        print("")

    elif args.load_from_pretrained:  #
        path = os.path.join(args.pretrained_model_dir, f"pretrain-epoch={args.pretraining_epoch_id}.ckpt")
        checkpoint_data = torch.load(path, map_location=load_map, weights_only=False)
        checkpoint_state = checkpoint_data['state_dict']
        model_state = model.state_dict()

        remapped_weights = {}
        for ckpt_key, ckpt_value in checkpoint_state.items():
            # Fix the redundant nesting: "model.encoder.encoder." → "model.encoder."
            if ckpt_key.startswith("model.encoder.encoder."):
                new_key = "model.encoder." + ckpt_key[len("model.encoder.encoder."):]
            else:
                new_key = ckpt_key

            # Match if key exists and shape is the same
            if new_key in model_state and model_state[new_key].shape == ckpt_value.shape:
                remapped_weights[new_key] = ckpt_value

        model.load_state_dict(remapped_weights, strict=False)

        print(f"Loaded pretrained weights from {path}")
        print(f"Matched weights: {len(remapped_weights)}/{len(model_state)} model parameters matched "
              f"(from {len(checkpoint_state)} pretrained parameters)")

    if torch.cuda.is_available():
        accelerator, devices, precision = "gpu", [args.gpu_id], "bf16-mixed"
    else:
        accelerator, devices, precision = "cpu", 1, "32-true"

    callbacks = []
    if checkpoint is not None:
        callbacks.append(checkpoint)
    if best_weights is not None:
        callbacks.append(best_weights)
    if early_stop is not None:
        callbacks.append(early_stop)
    callbacks.extend([tracker, TQDMProgressBar(refresh_rate=500)])

    trainer = pl.Trainer(
        default_root_dir=ckpt_dir,
        max_epochs=args.num_epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        precision=precision,
        devices=devices,
        num_sanity_val_steps=0,
        log_every_n_steps=max(1, min(50, len(train_loader))),
    )

    trainer.fit(model, train_loader, val_loader)

    final_weights_path = os.path.join(ckpt_dir, "final_state_dict.pt")
    try:
        torch.save(model.state_dict(), final_weights_path)
        print(f"Saved final model weights to {final_weights_path}")
    except OSError as exc:
        print(f"Warning: could not save final weights ({exc})")

    if checkpoint is not None:
        trainer.test(model, test_loader, ckpt_path="best")
    else:
        trainer.test(model, test_loader)

    if args.task_type == 'FD':
        plot_metrics(
            {"train_loss": tracker.losses["train_loss"], "val_loss": tracker.losses["val_loss"],
             "train_acc": tracker.accuracies["train_acc"], "val_acc": tracker.accuracies["val_acc"]},
            args.ckpt_dir,
            args.task_type
        )
    elif args.task_type == 'RUL':
        metric_name = model.regression_metric_name
        plot_metrics(
            {"train_loss": tracker.losses["train_loss"], "val_loss": tracker.losses["val_loss"],
             f"train_{metric_name}": tracker.rmses[f"train_{metric_name}"],
             f"val_{metric_name}": tracker.rmses[f"val_{metric_name}"]},
            args.ckpt_dir,
            args.task_type
        )

def apply_model_config(args):
    if getattr(args, "model_arch", "transformer") == "mlp":
        return
    config_map = {
        'tiny':  {'embed_dim': 128, 'heads': 4,  'depth': 4},
        'small': {'embed_dim': 256, 'heads': 8,  'depth': 8},
        'base':  {'embed_dim': 512, 'heads': 12, 'depth': 16},
    }
    config = config_map[args.model_type]
    for k, v in config.items():
        setattr(args, k, v)


TROLLEY_TARGET_NAMES = [
    "CSlot_Top_Gap",
    "CSlot_Length",
    "CSlot_Bot_Gap",
    "Latch_Height",
    "Latch_Length",
    "Nozzle_Diameter",
]


def apply_trolley_defaults(args):
    """Apply trolley regression defaults unless the user already set them."""
    if "trolley_regression" not in getattr(args, "data_id", ""):
        return
    if args.rul_target_names is None:
        args.rul_target_names = TROLLEY_TARGET_NAMES
    if args.input_normalize == "none":
        args.input_normalize = "standard"
    if args.target_normalize == "none":
        args.target_normalize = "standard"


def apply_mlp_notebook_defaults(args):
    """Match neural_network_regression_M1.ipynb training setup."""
    if getattr(args, "model_arch", "transformer") != "mlp":
        return
    args.num_epochs = 300
    args.batch_size = 32
    args.lr = 1e-3
    args.wt_decay = 1e-5
    args.mlp_dropout = 0.1
    args.input_normalize = "standard"
    args.target_normalize = "standard"
    args.load_from_pretrained = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default=r'./dataset/')
    parser.add_argument('--data_id', type=str, default=r'M01', help= 'choose [M01, M02,M03] for FD task and [FEMTO] for RUL task')
    parser.add_argument('--data_percentage', type=str, default="1")
    parser.add_argument('--model_id', type=str, default="CNC_FT", help= 'CNC_FT or FEMTO_FT')

    parser.add_argument('--model_arch', type=str, default='transformer', choices=['transformer', 'mlp'],
                        help='transformer: pretrained encoder backbone; mlp: skip encoder, flat 12-dim notebook MLP')
    parser.add_argument(
        '--regression_head',
        type=str,
        default='linear',
        choices=['linear', 'mlp'],
        help='Final layer on transformer pooled features: linear (default) or mlp (notebook 64->32 head)',
    )
    parser.add_argument('--model_type', type=str, choices=['tiny', 'small', 'base'], default='tiny')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--mlp_dropout', type=float, default=0.1, help='Dropout for MLP notebook architecture')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_moe', type=str2bool, default=False, help='[use MoE or default]')

    parser.add_argument('--load_from_pretrained', type=str2bool, default=True)
    parser.add_argument('--pretrained_model_dir', type=str, default="pretrained_models/Tiny")
    parser.add_argument('--pretraining_epoch_id', type=int, default=1)
    parser.add_argument('--pretrained_model_type', type=str, default='normal', help='model can be [normal, mae]')

    parser.add_argument('--num_epochs', type=int, default=600)
    parser.add_argument('--patience', type=int, default=50, help="For early stopping")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4) #1e-3
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--task_type',type=str,default='FD',choices=['FD', 'RUL'])
    parser.add_argument(
        '--rul_target_mode',
        type=str,
        default='auto',
        choices=['auto', 'single', 'multi'],
        help='RUL label handling: auto(infer), single(one target), multi(all targets, flattened)'
    )
    parser.add_argument(
        '--rul_single_target_index',
        type=int,
        default=0,
        help='Flattened index to use when --rul_target_mode single and labels have multiple dimensions'
    )
    parser.add_argument(
        '--rul_target_names',
        type=str,
        nargs='+',
        default=None,
        help='Optional names for multi-target regression outputs (used in plots/reports)'
    )
    parser.add_argument(
        '--input_normalize',
        type=str,
        default='none',
        choices=['none', 'zscore', 'standard'],
        help='Input normalization: none, zscore, or standard (notebook StandardScaler)'
    )
    parser.add_argument(
        '--target_normalize',
        type=str,
        default='none',
        choices=['none', 'standard'],
        help='Target normalization: none or standard (notebook StandardScaler on outputs)'
    )
    parser.add_argument(
        '--regression_loss',
        type=str,
        default='mae',
        choices=['mse', 'mae'],
        help='Regression training loss. Notebook trains with MSE on scaled targets; use mae for MAE loss.'
    )
    args = parser.parse_args()
    apply_trolley_defaults(args)
    apply_mlp_notebook_defaults(args)
    apply_model_config(args)
    main(args)

 # Tabular Regression   
# python fine_tune.py --task_type RUL --data_path C:\Users\ngyx\Desktop\Common_Model_framework\tabular_dataset --data_id regression_splits_S3 --data_percentage 1 --model_id regression_FT --model_type tiny --pretrained_model_dir pretrained_models/Tiny --pretraining_epoch_id 1 --batch_size 16 --num_epochs 100 --lr 3e-4 --gpu_id 0

# Tabular Classification
# python fine_tune.py --task_type FD --data_path C:\Users\ngyx\Desktop\Common_Model_framework\tabular_dataset --data_id classification_splits --data_percentage 1 --model_id classification_FT --model_type tiny --pretrained_model_dir pretrained_models/Tiny --pretraining_epoch_id 1 --batch_size 16 --num_epochs 100 --lr 3e-4 --gpu_id 0

# Time Series Regression
# python fine_tune.py --task_type RUL --data_path C:\Users\ngyx\Desktop\Common_Model_framework\time_series_dataset --data_id FEMTO_Regression/splits --data_percentage 1 --model_id FEMTO_FT --model_type tiny --pretrained_model_dir pretrained_models/Tiny --pretraining_epoch_id 1 --batch_size 16 --num_epochs 5 --lr 3e-4 --gpu_id 0

# Time series forecasting multi-target
# python fine_tune.py --task_type RUL --data_path time_series_dataset --data_id sanwa_forecasting/splits --data_percentage 1 --model_id sanwa_multi --model_type tiny --pretrained_model_dir pretrained_models/Tiny --pretraining_epoch_id 1 --batch_size 16 --num_epochs 1 --lr 3e-4 --gpu_id 0 --rul_target_mode multi

# time series forecasting single-target
# python fine_tune.py --task_type RUL --data_path time_series_dataset --data_id sanwa_forecasting/splits --data_percentage 1 --model_id sanwa_single --model_type tiny --pretrained_model_dir pretrained_models/Tiny --pretraining_epoch_id 1 --batch_size 16 --num_epochs 1 --lr 3e-4 --gpu_id 0 --rul_target_mode single --rul_single_target_index 0

# Time Series classification
# python fine_tune.py --task_type FD --data_path time_series_dataset --data_id taisin_time_series_classification --data_percentage 1 --model_id taisin_cclassification --model_type tiny --pretrained_model_dir pretrained_models/Tiny --pretraining_epoch_id 1 --batch_size 16 --num_epochs 2 --lr 3e-4 --gpu_id 0

# python fine_tune.py --task_type RUL --data_path tabular_dataset --data_id trolley_regression_splits --data_percentage 1 --model_id trolley_multi --rul_target_mode multi --input_normalize zscore --load_from_pretrained True --rul_target_names CSlot_Top_Gap CSlot_Length CSlot_Bot_Gap Latch_Height Latch_Length Nozzle_Diameter