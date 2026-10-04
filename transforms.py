"""Differentiable candle mathematics; importing this module performs no I/O."""
from __future__ import annotations

import math
import torch


def open_gap(open_price, previous_close):
    return torch.log(open_price / previous_close)


def body_return(open_price, close_price):
    return torch.log(close_price / open_price)


def upper_excursion(open_price, high_price, close_price):
    return torch.log(high_price / torch.maximum(open_price, close_price))


def lower_excursion(open_price, low_price, close_price):
    return torch.log(torch.minimum(open_price, close_price) / low_price)


def log_volume(volume):
    return torch.log1p(volume)


def log_returns(features):
    return features[..., 0] + features[..., 1]


def transform_ohlcv(raw, previous_close):
    """raw [...,T,5]; previous_close is the close immediately before raw."""
    raw = torch.as_tensor(raw)
    previous_close = torch.as_tensor(previous_close, dtype=raw.dtype, device=raw.device)
    if raw.ndim < 2 or raw.shape[-1] != 5 or raw.shape[-2] == 0:
        raise ValueError("OHLCV must be a nonempty [...,T,5] tensor")
    if not torch.isfinite(raw).all() or not torch.isfinite(previous_close).all():
        raise ValueError("Non-finite candle or anchor")
    o, h, l, c, v = raw.unbind(-1)
    if (raw[..., :4] <= 0).any() or (previous_close <= 0).any() or (v < 0).any():
        raise ValueError("Prices/anchor must be positive and volume nonnegative")
    if (h < torch.maximum(o, c)).any() or (l > torch.minimum(o, c)).any():
        raise ValueError("Invalid OHLC high/low relationship")
    anchor = torch.broadcast_to(previous_close, raw.shape[:-2]).unsqueeze(-1)
    previous = torch.cat((anchor, c[..., :-1]), dim=-1)
    return torch.stack((open_gap(o, previous), body_return(o, c),
                        upper_excursion(o, h, c), lower_excursion(o, l, c), log_volume(v)), -1)


def reconstruct_ohlcv(features, previous_close):
    """Exact recursive inverse; rejects invalid/nonfinite physical features."""
    if features.ndim < 2 or features.shape[-1] != 5 or features.shape[-2] == 0:
        raise ValueError("Features must be a nonempty [...,T,5] tensor")
    previous = torch.as_tensor(previous_close, dtype=features.dtype, device=features.device)
    previous = torch.broadcast_to(previous, features.shape[:-2])
    if not torch.isfinite(features).all() or not torch.isfinite(previous).all():
        raise ValueError("Non-finite features or anchor")
    if (features[..., 2:] < 0).any() or (previous <= 0).any():
        raise ValueError("Excursions/volume must be nonnegative and anchor positive")
    candles = []
    for token in features.unbind(-2):
        g, b, u, d, v = token.unbind(-1)
        o = previous * g.exp()
        c = o * b.exp()
        candles.append(torch.stack((o, torch.maximum(o, c) * u.exp(),
                                    torch.minimum(o, c) * (-d).exp(), c, torch.expm1(v)), -1))
        previous = c
    result = torch.stack(candles, -2)
    if not torch.isfinite(result).all() or (result[..., :4] <= 0).any():
        raise FloatingPointError("Reconstruction overflow/underflow; inspect predicted features")
    return result


def historical_volatility(context_features, ddof=0):
    return log_returns(context_features).std(dim=-1, correction=ddof)


def normalized_returns(returns, sigma, epsilon):
    return returns / (sigma.unsqueeze(-1) + epsilon)


def fit_threshold(training_normalized_returns, quantile):
    if training_normalized_returns.numel() == 0:
        raise ValueError("Threshold needs nonempty training targets")
    return float(torch.quantile(training_normalized_returns.abs().flatten(), quantile))


def trend_labels(z, threshold):
    labels = torch.ones_like(z, dtype=torch.long)
    return torch.where(z < -threshold, 0, torch.where(z > threshold, 2, labels))


def six_hour_return(returns):
    return returns.sum(-1)


def six_hour_normalized_return(returns, sigma, epsilon):
    return six_hour_return(returns) / (sigma * math.sqrt(returns.shape[-1]) + epsilon)
