"""Two-layer BiLSTM -> Bahdanau attention -> autoregressive LSTM."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = cfg.num_layers
        self.hidden = cfg.encoder_hidden_size
        self.lstm = nn.LSTM(cfg.input_size, self.hidden, self.layers, batch_first=True,
                            bidirectional=True, dropout=cfg.dropout)
        self.h_bridge = nn.ModuleList(nn.Linear(2 * self.hidden, cfg.decoder_hidden_size) for _ in range(self.layers))
        self.c_bridge = nn.ModuleList(nn.Linear(2 * self.hidden, cfg.decoder_hidden_size) for _ in range(self.layers))

    def bridge(self, state, projections):
        state = state.reshape(self.layers, 2, state.shape[1], self.hidden)
        joined = torch.cat((state[:, 0], state[:, 1]), -1)
        return torch.stack([torch.tanh(projections[i](joined[i])) for i in range(self.layers)])

    def forward(self, context):
        values, (h, c) = self.lstm(context)
        return values, (self.bridge(h, self.h_bridge), self.bridge(c, self.c_bridge))


class BahdanauAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.query = nn.Linear(cfg.decoder_hidden_size, cfg.attention_dim, bias=False)
        self.values = nn.Linear(2 * cfg.encoder_hidden_size, cfg.attention_dim, bias=False)
        self.energy = nn.Linear(cfg.attention_dim, 1, bias=False)

    def forward(self, query, values):
        scores = self.energy(torch.tanh(self.query(query)[:, None] + self.values(values))).squeeze(-1)
        weights = scores.softmax(-1)
        return torch.bmm(weights[:, None], values).squeeze(1), weights


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = 2 * cfg.encoder_hidden_size
        self.lstm = nn.LSTM(cfg.input_size + width, cfg.decoder_hidden_size, cfg.num_layers,
                            batch_first=True, dropout=cfg.dropout)
        self.head = nn.Sequential(nn.Linear(cfg.decoder_hidden_size + width, cfg.head_hidden_size),
                                  nn.GELU(), nn.Linear(cfg.head_hidden_size, cfg.input_size))

    def forward(self, token, context, state):
        out, state = self.lstm(torch.cat((token, context), -1)[:, None], state)
        representation = torch.cat((out[:, 0], context), -1)
        return self.head(representation), representation, state


class seq2seq_model(nn.Module):
    def __init__(self, cfg, scaler, experiment, initialization_seed=None):
        super().__init__()
        if experiment not in ("2A", "2B"):
            raise ValueError("Experiment must be 2A or 2B")
        self.cfg, self.experiment = cfg, experiment
        self.initialization_seed = cfg.seed if initialization_seed is None else initialization_seed
        # CPU module initialization owns a seeded stream, independent of caller RNG.
        # fork_rng restores the caller stream; manual_seed on this generator does
        # not seed or advance any CUDA generator.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.initialization_seed)
            self.encoder = Encoder(cfg)
            self.attention = BahdanauAttention(cfg)
            self.decoder = Decoder(cfg)
        self.register_buffer("feature_mean", scaler.mean.float().clone())
        self.register_buffer("feature_scale", scaler.scale.float().clone())
        # Bound must be valid under both the original float64 fold scaler and
        # the model's float32 copy. Round toward the physical interior, not below
        # zero when softplus underflows; the adjustment is only rounding error.
        exact_lower = torch.maximum(-scaler.mean[2:].double() / scaler.scale[2:].double(),
                                    -self.feature_mean[2:].double() / self.feature_scale[2:].double())
        lower = exact_lower.float()
        lower = torch.where(lower.double() < exact_lower,
                            torch.nextafter(lower, torch.full_like(lower, float('inf'))), lower)
        self.register_buffer("positive_lower", lower)
        ratio = (self.feature_mean[2:] / self.feature_scale[2:]).clamp_min(cfg.epsilon)
        self.register_buffer("positive_bias", ratio + torch.log(-torch.expm1(-ratio)))
        self.trend_head = None
        if experiment == "2B":
            # Restore global RNG after head initialization: shared initialization,
            # subsequent dropout, and teacher-forcing streams stay comparable.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self.initialization_seed + cfg.head_seed_offset)
                self.trend_head = nn.Sequential(
                    nn.Linear(cfg.decoder_hidden_size + 2 * cfg.encoder_hidden_size, cfg.head_hidden_size),
                    nn.GELU(), nn.Linear(cfg.head_hidden_size, 3))

    def physical(self, predictions):
        return predictions * self.feature_scale + self.feature_mean

    def constrain(self, output):
        # Algebraically equivalent to the previous scale-aware physical softplus,
        # but keeps signed channels exact and avoids the redundant round trip.
        positive = self.positive_lower + F.softplus(output[..., 2:] + self.positive_bias)
        return torch.cat((output[..., :2], positive), -1)

    def forward(self, context, targets=None, teacher_forcing=0.0, generator=None):
        if context.ndim != 3 or tuple(context.shape[1:]) != (48, 5):
            raise ValueError("Context must be [B,48,5]")
        if not 0 <= teacher_forcing <= 1:
            raise ValueError("Teacher forcing must be between zero and one")
        if targets is not None and tuple(targets.shape) != (len(context), 12, 5):
            raise ValueError("Targets must be [B,12,5]")
        if teacher_forcing and (targets is None or not self.training):
            raise ValueError("Teacher forcing requires training mode and targets")
        values, state = self.encoder(context)
        token = context[:, -1]
        predictions, attention, logits = [], [], []
        for step in range(self.cfg.horizon):
            attended, weights = self.attention(state[0][-1], values)
            raw_output, representation, state = self.decoder(token, attended, state)
            token = self.constrain(raw_output)
            predictions.append(token); attention.append(weights)
            if self.trend_head is not None:
                logits.append(self.trend_head(representation))
            if step < self.cfg.horizon - 1 and teacher_forcing:
                use_truth = torch.rand((len(context), 1), generator=generator,
                                       device=context.device) < teacher_forcing
                token = torch.where(use_truth, targets[:, step], token)
        return {"predictions": torch.stack(predictions, 1), "attention": torch.stack(attention, 1),
                "trend_logits": torch.stack(logits, 1) if logits else None}
