"""Chronological partitions, training-only statistics, and forecast windows."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from transforms import (fit_threshold, historical_volatility, log_returns,
                        normalized_returns, transform_ohlcv, trend_labels)

RAW_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


def load_dataset(path, cfg):
    frame = pd.read_csv(path)
    if list(frame.columns) != RAW_COLUMNS:
        raise ValueError(f"Expected columns {RAW_COLUMNS}")
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    for column in RAW_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.isna().any().any() or not np.isfinite(frame.iloc[:, 1:].to_numpy()).all():
        raise ValueError("Dataset contains missing/non-finite values")
    times = frame.datetime
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and chronological")
    if not times.diff().iloc[1:].eq(pd.Timedelta(minutes=cfg.candle_minutes)).all():
        raise ValueError("Dataset must have uninterrupted 30-minute cadence")
    # Validate every row, including the first (anchor here is only for validation).
    transform_ohlcv(torch.tensor(frame.iloc[:, 1:].to_numpy(), dtype=torch.float64), frame.close.iloc[0])
    return frame


@dataclass(frozen=True)
class Fold:
    index: int
    train_end: int
    validation_start: int
    validation_end: int


def chronological_split(row_count, cfg):
    test_rows = math.floor(row_count * cfg.test_fraction / cfg.horizon) * cfg.horizon
    dev_end = row_count - test_rows
    validation_rows = (dev_end // cfg.horizon // 2 // cfg.n_folds) * cfg.horizon
    initial = dev_end - cfg.n_folds * validation_rows
    if min(initial - 1, validation_rows, test_rows) < cfg.context_length + cfg.horizon:
        raise ValueError("Partitions too short for 48-context/12-target windows")
    folds = [Fold(i + 1, initial + i * validation_rows, initial + i * validation_rows,
                  initial + (i + 1) * validation_rows) for i in range(cfg.n_folds)]
    return dev_end, folds


@dataclass
class Standardizer:
    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, features, epsilon):
        mean = features.mean(0)
        scale = features.std(0, correction=0)
        # Constant channels are valid (e.g. zero gaps). Unit scale preserves them.
        scale = torch.where(scale > epsilon, scale, torch.ones_like(scale))
        return cls(mean, scale)

    def transform(self, features):
        return (features - self.mean.to(features)) / self.scale.to(features)

    def inverse(self, features):
        return features * self.scale.to(features) + self.mean.to(features)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


class CandleWindows(Dataset):
    def __init__(self, frame, start, end, cfg, stride, scaler=None, threshold=None):
        self.cfg = cfg
        self.frame = frame
        self.partition_start, self.partition_end = start, end
        raw = torch.tensor(frame.iloc[start:end, 1:].to_numpy(), dtype=torch.float64)
        self.offset = 1 if start == 0 else 0
        self.raw = raw
        self.features = transform_ohlcv(raw[self.offset:],
                                       float(frame.close.iloc[start - 1 if start else 0]))
        self.scaler = scaler or Standardizer.fit(self.features, cfg.epsilon)
        self.standardized = self.scaler.transform(self.features).float()
        available = len(self.features) - cfg.context_length - cfg.horizon
        self.starts = torch.arange(0, available + 1, stride) if available >= 0 else torch.empty(0, dtype=torch.long)
        if not len(self.starts):
            raise ValueError("Partition has no complete forecast windows")
        returns = log_returns(self.features)
        context_indices = self.starts[:, None] + torch.arange(cfg.context_length)
        target_indices = self.starts[:, None] + cfg.context_length + torch.arange(cfg.horizon)
        self.sigmas = returns[context_indices].std(-1, correction=cfg.volatility_ddof)
        self.z = normalized_returns(returns[target_indices], self.sigmas, cfg.epsilon)
        self.threshold = fit_threshold(self.z, cfg.threshold_quantile) if threshold is None else threshold
        self.labels = trend_labels(self.z, self.threshold)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        s = int(self.starts[index]); t = s + self.cfg.context_length
        raw_t = t + self.offset
        return {"context": self.standardized[s:t], "target": self.standardized[t:t + self.cfg.horizon],
                "raw_target": self.raw[raw_t:raw_t + self.cfg.horizon],
                "raw_context": self.raw[s + self.offset:raw_t],
                "reference_close": self.raw[raw_t - 1, 3], "sigma": self.sigmas[index],
                "labels": self.labels[index], "origin_index": self.partition_start + raw_t - 1,
                "target_indices": torch.arange(self.partition_start + raw_t,
                                                self.partition_start + raw_t + self.cfg.horizon)}


def fold_datasets(frame, fold, cfg):
    train = CandleWindows(frame, 0, fold.train_end, cfg, cfg.train_stride)
    valid = CandleWindows(frame, fold.validation_start, fold.validation_end, cfg, cfg.eval_stride,
                          train.scaler, train.threshold)
    return train, valid


def final_datasets(frame, dev_end, cfg):
    train = CandleWindows(frame, 0, dev_end, cfg, cfg.train_stride)
    test = CandleWindows(frame, dev_end, len(frame), cfg, cfg.eval_stride, train.scaler, train.threshold)
    return train, test


def loader(dataset, cfg, shuffle=False, seed=None):
    generator = torch.Generator().manual_seed(cfg.seed if seed is None else seed)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, generator=generator)


def describe_split(frame, cfg):
    dev_end, folds = chronological_split(len(frame), cfg)
    return {"rows": len(frame), "start": str(frame.datetime.iloc[0]), "end": str(frame.datetime.iloc[-1]),
            "development_end_exclusive": dev_end, "test_rows": len(frame) - dev_end,
            "folds": [asdict(f) for f in folds],
            "train_stride": cfg.train_stride, "eval_stride": cfg.eval_stride,
            "evaluation_context_policy": "48 context candles inside each validation/test partition"}
