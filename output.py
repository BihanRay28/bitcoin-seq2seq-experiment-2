"""Progress, transparent metrics, histories, and forecast artifacts."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transforms import (log_returns, reconstruct_ohlcv, normalized_returns,
                        six_hour_normalized_return, trend_labels)

CLASSES = ["Bear", "Neutral", "Bull"]
PERSISTENCE_DEFINITION = {
    'open': 'last observed close C_t at every forecast step',
    'high': 'last observed close C_t at every forecast step',
    'low': 'last observed close C_t at every forecast step',
    'close': 'last observed close C_t at every forecast step',
    'volume': 'last observed volume V_t at every forecast step',
    'log_returns': 'zero at every step; six-hour cumulative return is zero',
    'trend': 'Neutral (class 1) at every step',
}


def persistence_forecast(raw_context, horizon):
    """Causal baseline: takes observed candles only, never future targets."""
    context = np.asarray(raw_context)
    forecast = np.empty((len(context), horizon, 5), dtype=context.dtype)
    forecast[..., :4] = context[:, -1, 3, None, None]
    forecast[..., 4] = context[:, -1, 4, None]
    return forecast


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def create_run(output_root, experiment, mode):
    run_id = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%d_%H%M%S_%f") + "_" + mode
    directories = {}
    for name in ("curves", "matrices", "history", "final reconstruction testing"):
        path = Path(output_root) / experiment / name / run_id
        path.mkdir(parents=True, exist_ok=False)
        directories[name] = path
    return run_id, directories


def progress(loader, description):
    return tqdm(loader, desc=description, leave=False, dynamic_ncols=True, unit="batch")


def classification(actual, predicted):
    actual, predicted = np.asarray(actual).ravel(), np.asarray(predicted).ravel()
    cm = np.bincount(3 * actual.astype(int) + predicted.astype(int), minlength=9).reshape(3, 3)
    return classification_matrix(cm)


def classification_matrix(cm):
    cm = np.asarray(cm)
    support, predicted_count = cm.sum(1), cm.sum(0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(3), where=support > 0)
    precision = np.divide(cm.diagonal(), predicted_count, out=np.zeros(3), where=predicted_count > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=precision + recall > 0)
    return {"accuracy": float(cm.trace() / max(1, cm.sum())), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall[support > 0].mean()) if (support > 0).any() else 0.0,
            "confusion_matrix": cm.tolist(), "actual_class_counts": support.tolist(),
            "predicted_class_counts": predicted_count.tolist(), "per_class_f1": f1.tolist(),
            "class_order": CLASSES,
            "actual_class_proportions": (support / max(1, support.sum())).tolist(),
            "predicted_class_proportions": (predicted_count / max(1, predicted_count.sum())).tolist()}


def regression(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt((error ** 2).mean()))}


def forecast_metrics(arrays, threshold, cfg):
    predicted, actual = arrays["physical_predictions"], arrays["physical_targets"]
    pr = predicted[..., 0] + predicted[..., 1]; ar = actual[..., 0] + actual[..., 1]
    sigma = arrays["sigma"]
    az = ar / (sigma[:, None] + cfg.epsilon)
    pz = pr / (sigma[:, None] + cfg.epsilon)
    labels = lambda values: np.where(values < -threshold, 0, np.where(values > threshold, 2, 1))
    truth, pred = labels(az), labels(pz)
    metrics = {"derived_trend": classification(truth, pred),
               "returns": regression(ar, pr), "six_hour_returns": regression(ar.sum(1), pr.sum(1)),
               "six_hour_direction_accuracy": float(((ar.sum(1) > 0) == (pr.sum(1) > 0)).mean()),
               "actual_return_std": float(ar.std()), "predicted_return_std": float(pr.std())}
    denominator = sigma * np.sqrt(cfg.horizon) + cfg.epsilon
    metrics["six_hour_trend"] = classification(labels(ar.sum(1) / denominator), labels(pr.sum(1) / denominator))
    predicted_raw, actual_raw = arrays["raw_predictions"], arrays["raw_targets"]
    persistence = persistence_forecast(arrays['raw_context'], cfg.horizon)
    metrics['persistence_definition'] = PERSISTENCE_DEFINITION
    metrics["ohlcv"] = {name: regression(actual_raw[..., i], predicted_raw[..., i])
                        for i, name in enumerate(("open", "high", "low", "close", "volume"))}
    metrics["persistence_ohlcv"] = {name: regression(actual_raw[..., i], persistence[..., i])
                                    for i, name in enumerate(("open", "high", "low", "close", "volume"))}
    metrics["persistence_returns"] = regression(ar, np.zeros_like(ar))
    metrics["persistence_six_hour_returns"] = regression(ar.sum(1), np.zeros(len(ar)))
    metrics["persistence_trend"] = classification(truth, np.ones_like(truth))
    metrics["per_horizon"] = [{"step": j + 1, **regression(ar[:, j], pr[:, j]),
                                "trend_accuracy": float((truth[:, j] == pred[:, j]).mean())}
                               for j in range(cfg.horizon)]
    if "trend_logits" in arrays:
        head_pred = arrays["trend_logits"].argmax(-1)
        metrics["classification_head"] = classification(truth, head_pred)
        metrics["head_return_disagreement"] = float((head_pred != pred).mean())
    return metrics


def epoch_summary(record, label):
    validation = (f"val={record['validation_total']:.5f} val_acc={record['validation_accuracy']:.3f} "
                  f"val_F1={record['validation_macro_f1']:.3f} " if 'validation_total' in record else '')
    tqdm.write(f"{label} epoch={record['epoch']} stage={record['stage'] + 1} "
               f"stage_epoch={record['stage_epoch']} TF={record['teacher_forcing']:.2f} "
               f"LR={record['learning_rate']:.6g} train={record['train_total']:.5f} "
               f"{validation}"
               f"grad={record['gradient_norm_before']:.4f}/{record['gradient_norm_after']:.4f} "
               f"patience={record.get('stale_epochs', 0)} seconds={record['seconds']:.1f}")


def aggregate_folds(fold_results, directories):
    """Pool fold counts, not rounded accuracies; save fold-to-fold comparison."""
    summary = {'fold_count': len(fold_results), 'samples': sum(r['metrics']['samples'] for r in fold_results)}
    for key in ('derived_trend', 'six_hour_trend', 'classification_head', 'persistence_trend'):
        available = [r['metrics'][key] for r in fold_results if key in r['metrics']]
        if available:
            summary[key] = classification_matrix(np.sum([r['confusion_matrix'] for r in available], axis=0))
    for key in ('returns', 'six_hour_returns', 'persistence_returns', 'persistence_six_hour_returns'):
        weights = np.array([r['metrics']['samples'] for r in fold_results], dtype=float)
        summary[key] = {'mae': float(np.average([r['metrics'][key]['mae'] for r in fold_results], weights=weights)),
                        'rmse': float(np.sqrt(np.average([r['metrics'][key]['rmse']**2 for r in fold_results], weights=weights)))}
    confusion_charts(summary, directories, 'walk_forward_aggregate')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ids = [r['fold'] for r in fold_results]
    axes[0].plot(ids, [r['metrics']['derived_trend']['macro_f1'] for r in fold_results], marker='o')
    axes[0].set_ylabel('Derived trend macro-F1')
    axes[1].plot(ids, [r['metrics']['returns']['mae'] for r in fold_results], marker='o', label='Model')
    axes[1].plot(ids, [r['metrics']['persistence_returns']['mae'] for r in fold_results], marker='o', label='Persistence')
    axes[1].set_ylabel('Return MAE'); axes[1].legend()
    for ax in axes: ax.set_xlabel('Walk-forward fold'); ax.set_xticks(ids); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(directories['curves']/'walk_forward_comparison.png', dpi=150); plt.close(fig)
    return summary


def save_history(history, gradient_history, directories, label):
    folder = directories["history"] / label
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "epochs.json", history)
    pd.DataFrame(history).to_csv(folder / "epochs.csv", index=False)
    write_json(folder / "parameter_gradient_history.json", gradient_history)
    curves = directories["curves"] / label
    curves.mkdir(parents=True, exist_ok=True)
    groups = {
        "losses": ["train_total", "validation_total", "train_base", "validation_base",
                   "train_return", "validation_return", "train_trend", "validation_trend"],
        "trend_metrics": ["validation_accuracy", "validation_macro_f1", "validation_balanced_accuracy",
                          "validation_head_accuracy", "validation_head_macro_f1"],
        "return_error": ["validation_return_mae", "validation_six_hour_return_mae"],
        "teacher_forcing": ["teacher_forcing"],
        "gradient_norms": ["gradient_norm_before", "gradient_norm_after"],
    }
    for name, fields in groups.items():
        fig, ax = plt.subplots(figsize=(9, 4))
        present = [field for field in fields if any(field in row for row in history)]
        if not present:
            plt.close(fig); continue
        for field in present:
            ax.plot([r["epoch"] for r in history], [r.get(field, np.nan) for r in history], label=field)
        for a, b in zip(history, history[1:]):
            if a["stage"] != b["stage"]:
                ax.axvline(b["epoch"], color="gray", linestyle=":", alpha=.5)
        ax.set(title=f"{label}: {name.replace('_', ' ')}", xlabel="Epoch")
        ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(curves / f"{name}.png", dpi=150); plt.close(fig)


def confusion_charts(metrics, directories, label):
    folder = directories["matrices"] / label
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("derived_trend", "six_hour_trend", "classification_head", "persistence_trend"):
        if name not in metrics:
            continue
        counts = np.asarray(metrics[name]["confusion_matrix"])
        pd.DataFrame(counts, index=CLASSES, columns=CLASSES).to_csv(folder / f"{name}_counts.csv")
        for normalized in (False, True):
            matrix = counts.astype(float)
            if normalized:
                matrix = np.divide(matrix, matrix.sum(1, keepdims=True), out=np.zeros_like(matrix),
                                   where=matrix.sum(1, keepdims=True) > 0)
            fig, ax = plt.subplots(figsize=(5, 4)); im = ax.imshow(matrix, cmap="Blues")
            ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
                   xlabel="Predicted", ylabel="Actual", title=f"{label}: {name}")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{matrix[i,j]:.2f}" if normalized else str(counts[i,j]), ha="center", va="center")
            fig.colorbar(im, ax=ax); fig.tight_layout()
            fig.savefig(folder / f"{name}_{'normalized' if normalized else 'counts'}.png", dpi=150)
            plt.close(fig)


def candle_panel(ax, values, start=0, alpha=1):
    for j, (o, h, l, c, _) in enumerate(values):
        x = j + start; color = "#168866" if c >= o else "#c34d58"
        ax.vlines(x, l, h, color=color, alpha=alpha)
        if c == o:
            ax.hlines(c, x - .3, x + .3, color=color, alpha=alpha)
        else:
            ax.add_patch(Rectangle((x - .3, min(o, c)), .6, abs(c - o), color=color, alpha=alpha))
    ax.autoscale_view(); ax.grid(alpha=.2)


def save_forecasts(arrays, metrics, frame, directories, label, cfg):
    folder = directories["final reconstruction testing"] / label
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "predictions_and_attention.npz", **arrays)
    rows = []
    for n, indices in enumerate(arrays["target_indices"]):
        for step, index in enumerate(indices):
            row = {"origin_index": int(arrays["origin_index"][n]), "origin_time": str(frame.datetime.iloc[int(arrays['origin_index'][n])]),
                   "timestamp": str(frame.datetime.iloc[int(index)]), "step": step + 1}
            for i, name in enumerate(("open", "high", "low", "close", "volume")):
                row[f"actual_{name}"] = float(arrays["raw_targets"][n, step, i])
                row[f"predicted_{name}"] = float(arrays["raw_predictions"][n, step, i])
            rows.append(row)
    pd.DataFrame(rows).to_csv(folder / "actual_predicted_ohlcv.csv", index=False)
    write_json(folder / "metrics.json", metrics)
    # Last complete window retains its own observed context and forecast origin.
    n = len(arrays["raw_predictions"]) - 1
    context = arrays["raw_context"][n]
    fig, axes = plt.subplots(2, 2, figsize=(16, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for col, kind in enumerate(("raw_targets", "raw_predictions")):
        combined = np.concatenate((context, arrays[kind][n]))
        candle_panel(axes[0, col], context, alpha=.4)
        candle_panel(axes[0, col], arrays[kind][n], start=len(context))
        axes[0, col].axvline(len(context) - .5, color="black", linestyle="--")
        axes[0, col].set_title(f"{label}: {'Actual' if col == 0 else 'Predicted'}; 48 observed + 12 future")
        axes[1, col].bar(range(len(combined)), combined[:, 4], color="#536a83")
        axes[1, col].set_ylabel("Volume")
        origin = int(arrays["origin_index"][n]); indices = np.arange(origin - 47, origin + 13)
        ticks = list(range(0, len(combined), 12)) + [len(combined) - 1]
        axes[1, col].set_xticks(ticks, [frame.datetime.iloc[int(indices[t])].strftime("%m-%d\n%H:%M") for t in ticks])
    axes[0, 0].set_ylabel("Price (USDT)")
    low = min(np.concatenate((context, arrays["raw_targets"][n], arrays["raw_predictions"][n]))[:, 2])
    high = max(np.concatenate((context, arrays["raw_targets"][n], arrays["raw_predictions"][n]))[:, 1])
    for ax in axes[0]: ax.set_ylim(low - (high-low)*.05, high + (high-low)*.05)
    fig.tight_layout(); fig.savefig(folder / "last_window_candles.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, indices in enumerate(arrays["target_indices"]):
        times = frame.datetime.iloc[indices].to_numpy()
        ax.plot(times, arrays["raw_targets"][i, :, 3], color="#168866", label="Actual" if i == 0 else None)
        ax.plot(times, arrays["raw_predictions"][i, :, 3], color="#c34d58", alpha=.65,
                label="Predicted (each window has its own observed anchor)" if i == 0 else None)
    ax.set(title=f"{label}: close-price forecasts", ylabel="USDT", xlabel="Dataset timestamp")
    ax.legend(); ax.grid(alpha=.2); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(folder / "period_close_comparison.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    horizon = metrics["per_horizon"]
    axes[0].plot([x['step'] for x in horizon], [x['mae'] for x in horizon]); axes[0].set_title("Return MAE")
    axes[1].plot([x['step'] for x in horizon], [x['trend_accuracy'] for x in horizon]); axes[1].set_title("Derived trend accuracy")
    for ax in axes: ax.set_xlabel("Forecast step"); ax.grid(alpha=.2)
    fig.tight_layout(); (directories["curves"] / label).mkdir(parents=True, exist_ok=True)
    fig.savefig(directories["curves"] / label / "horizon_metrics.png", dpi=150); plt.close(fig)


class GradientHistory:
    """Aggregate tensor statistics on-device, transfer once per epoch."""
    def __init__(self):
        self.entries = {}

    def add(self, model):
        for name, parameter in model.named_parameters():
            entry = self.entries.setdefault(name, {"batches": 0, "missing": 0, "stats": None})
            entry["batches"] += 1
            if parameter.grad is None:
                entry["missing"] += 1; continue
            grad = parameter.grad.detach().double()
            if not torch.isfinite(grad).all():
                raise FloatingPointError(f"Nonfinite gradient: {name}")
            # Sum and squared sum allow pooled mean/std across all batches.
            stats = torch.stack((grad.sum(), grad.square().sum(), grad.min(), grad.max(),
                                 grad.norm(), grad.new_tensor(grad.numel())))
            if entry["stats"] is None: entry["stats"] = stats
            else:
                previous = entry["stats"]
                entry["stats"] = torch.stack((previous[0] + stats[0], previous[1] + stats[1],
                                              torch.minimum(previous[2], stats[2]), torch.maximum(previous[3], stats[3]),
                                              previous[4] + stats[4], previous[5] + stats[5]))

    def finish(self, model):
        result = {}
        for name, parameter in model.named_parameters():
            value = parameter.detach().double()
            weight = torch.stack((value.mean(), value.std(correction=0), value.min(), value.max(), value.norm())).cpu().tolist()
            entry = self.entries[name]
            item = {"parameter": dict(zip(("mean", "std", "min", "max", "norm"), weight)),
                    "batches": entry['batches'], "missing_gradient_batches": entry['missing'], "gradient": None}
            item['missing_gradient_status'] = ('all' if entry['missing'] == entry['batches'] else
                                               'some' if entry['missing'] else 'none')
            if entry['stats'] is not None:
                total, squares, lo, hi, norms, count = entry['stats'].cpu().tolist()
                mean = total / count
                item['gradient'] = {"mean": mean, "std": float(np.sqrt(max(0, squares / count - mean*mean))),
                                    "min": lo, "max": hi, "mean_batch_norm": norms / (entry['batches'] - entry['missing']),
                                    "norm": float(np.sqrt(squares)),
                                    "norm_definition": 'L2 norm pooled over pre-clipping gradients of all present batches'}
            result[name] = item
        return result
