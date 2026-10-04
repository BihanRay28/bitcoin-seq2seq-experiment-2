"""Experiment objectives and staged teacher-forcing training."""
from __future__ import annotations

import math
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from transforms import log_returns, normalized_returns, reconstruct_ohlcv, transform_ohlcv
from output import GradientHistory, epoch_summary, forecast_metrics, progress, save_history, write_json


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def losses(model, result, batch, cfg, weights):
    prediction = result['predictions']; target = batch['target'].to(prediction)
    base = F.smooth_l1_loss(prediction, target, beta=cfg.smooth_l1_beta)
    zero = base.new_zeros(())
    if model.experiment == '2A':
        return {'base': base, 'return': zero, 'trend': zero, 'total': base}
    physical_prediction, physical_target = model.physical(prediction), model.physical(target)
    sigma = batch['sigma'].to(prediction)
    ret = F.smooth_l1_loss(normalized_returns(log_returns(physical_prediction), sigma, cfg.epsilon),
                           normalized_returns(log_returns(physical_target), sigma, cfg.epsilon), beta=cfg.smooth_l1_beta)
    trend = F.cross_entropy(result['trend_logits'].reshape(-1, 3), batch['labels'].to(prediction.device).reshape(-1))
    total = base + weights['return'] * ret + weights['trend'] * trend
    return {'base': base, 'return': ret, 'trend': trend, 'total': total}


def calibrate(model, loader, cfg, device):
    if model.experiment == '2A':
        return {'return': 0.0, 'trend': 0.0, 'calibration': None}
    model.eval(); sums = {key: 0.0 for key in ('base', 'return', 'trend')}; count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= cfg.calibration_batches: break
            context = batch['context'].to(device)
            values = losses(model, model(context), batch, cfg, {'return': 1., 'trend': 1.})
            for key in sums: sums[key] += float(values[key]) * len(context)
            count += len(context)
    if not count: raise ValueError('No training calibration samples')
    means = {key: value / count for key, value in sums.items()}
    if not all(math.isfinite(x) for x in means.values()): raise FloatingPointError('Nonfinite calibration')
    return {'return': cfg.return_contribution * means['base'] / max(means['return'], cfg.epsilon),
            'trend': cfg.trend_contribution * means['base'] / max(means['trend'], cfg.epsilon),
            'calibration': {'samples': count, 'batches_limit': cfg.calibration_batches, 'means': means}}


@torch.no_grad()
def evaluate(model, loader, cfg, weights, device, threshold):
    model.eval(); sums = dict.fromkeys(('base', 'return', 'trend', 'total'), 0.0); count = 0; chunks = {}
    for batch in loader:
        context = batch['context'].to(device)
        result = model(context)
        values = losses(model, result, batch, cfg, weights)
        if not all(torch.isfinite(v) for v in values.values()): raise FloatingPointError('Nonfinite validation/test loss')
        for key in sums: sums[key] += float(values[key]) * len(context)
        count += len(context)
        physical = model.physical(result['predictions']).double()
        physical_target = transform_ohlcv(batch['raw_target'].to(device), batch['reference_close'].to(device))
        raw = reconstruct_ohlcv(physical, batch['reference_close'].to(device))
        payload = {'physical_predictions': physical, 'physical_targets': physical_target,
                   'raw_predictions': raw, 'attention': result['attention']}
        if result['trend_logits'] is not None: payload['trend_logits'] = result['trend_logits']
        for key in ('raw_target', 'raw_context', 'reference_close', 'sigma', 'origin_index', 'target_indices'):
            payload['raw_targets' if key == 'raw_target' else key] = batch[key]
        for key, value in payload.items(): chunks.setdefault(key, []).append(value.detach().cpu().numpy())
    if not count: raise ValueError('Empty evaluation loader')
    arrays = {key: np.concatenate(value) for key, value in chunks.items()}
    metrics = forecast_metrics(arrays, threshold, cfg)
    metrics['losses'] = {key: value/count for key, value in sums.items()}
    metrics['samples'] = count
    return metrics, arrays


def train_epoch(model, loader, optimizer, cfg, weights, device, probability, generator, label):
    model.train(); collector = GradientHistory(); sums = dict.fromkeys(('base', 'return', 'trend', 'total'), 0.0)
    count = 0; before_sum = after_sum = 0.0; batches = 0
    bar = progress(loader, label)
    for batch in bar:
        context, target = batch['context'].to(device), batch['target'].to(device)
        optimizer.zero_grad(set_to_none=True)
        result = model(context, target, teacher_forcing=probability, generator=generator)
        values = losses(model, result, batch, cfg, weights)
        if not all(torch.isfinite(v) for v in values.values()): raise FloatingPointError('Nonfinite training loss')
        values['total'].backward(); collector.add(model)
        norm_before = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip, error_if_nonfinite=True))
        norm_after = float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])))
        optimizer.step()
        for key in sums: sums[key] += float(values[key].detach()) * len(context)
        count += len(context); batches += 1; before_sum += norm_before; after_sum += norm_after
        bar.set_postfix(loss=f"{sums['total']/count:.5f}", base=f"{sums['base']/count:.5f}",
                        ret=f"{sums['return']/count:.5f}", trend=f"{sums['trend']/count:.5f}", tf=f"{probability:.2f}")
    if not count: raise ValueError('Empty training loader')
    return {key: value/count for key, value in sums.items()}, collector.finish(model), before_sum/batches, after_sum/batches


def checkpoint(model, optimizer, cfg, weights, threshold, epoch, stage, stage_epoch):
    return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
            'stage': stage, 'stage_epoch': stage_epoch, 'config': vars(cfg), 'loss_weights': weights,
            'threshold': threshold, 'scaler': {'mean': model.feature_mean.cpu().tolist(), 'scale': model.feature_scale.cpu().tolist()}}


def restore(path, model, optimizer, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
    return state


def fit(model, train_loader, valid_loader, cfg, weights, device, threshold, directories, label, fixed_durations=None):
    optimizer = make_optimizer(model, cfg)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + cfg.teacher_seed_offset)
    folder = directories['history'] / label; folder.mkdir(parents=True, exist_ok=True)
    history, gradients, durations, selected_stage_epochs = [], [], [], []; epoch = 0
    for stage, probability in enumerate(cfg.teacher_probabilities):
        cap = fixed_durations[stage] if fixed_durations is not None else cfg.stage_max_epochs[stage]
        patience = cfg.stage_patience[stage]; best = float('inf'); stale = 0; best_stage_epoch = 0
        stage_checkpoint = folder / f'stage_{stage+1:02d}_best.pt'
        tqdm.write(f'{label}: stage {stage+1}, teacher forcing={probability}, cap={cap}, patience={patience}')
        for stage_epoch in range(1, cap + 1):
            epoch += 1; started = time.perf_counter()
            values, stats, before, after = train_epoch(model, train_loader, optimizer, cfg, weights, device,
                                                      probability, generator, f'{label} S{stage+1} E{epoch}')
            record = {'epoch': epoch, 'stage': stage, 'stage_epoch': stage_epoch, 'teacher_forcing': probability,
                      'learning_rate': optimizer.param_groups[0]['lr'], **{f'train_{k}': v for k, v in values.items()},
                      'gradient_norm_before': before, 'gradient_norm_after': after}
            if valid_loader is not None:
                metrics, _ = evaluate(model, valid_loader, cfg, weights, device, threshold)
                record.update({f'validation_{k}': v for k, v in metrics['losses'].items()})
                record.update({'validation_accuracy': metrics['derived_trend']['accuracy'],
                               'validation_macro_f1': metrics['derived_trend']['macro_f1'],
                               'validation_balanced_accuracy': metrics['derived_trend']['balanced_accuracy'],
                               'validation_return_mae': metrics['returns']['mae'],
                               'validation_six_hour_return_mae': metrics['six_hour_returns']['mae']})
                if 'classification_head' in metrics:
                    record.update({'validation_head_accuracy': metrics['classification_head']['accuracy'],
                                   'validation_head_macro_f1': metrics['classification_head']['macro_f1']})
                score = metrics['losses']['total']
                if score < best - cfg.min_delta:
                    best = score; stale = 0; best_stage_epoch = stage_epoch
                    torch.save(checkpoint(model, optimizer, cfg, weights, threshold, epoch, stage, stage_epoch), stage_checkpoint)
                else: stale += 1
            record['stale_epochs'] = stale; record['seconds'] = time.perf_counter() - started
            history.append(record); gradients.append({'epoch': epoch, 'stage': stage, 'parameters': stats})
            # Incremental histories survive interruptions; plots are generated at fit completion.
            write_json(folder / 'epochs.json', history); write_json(folder / 'parameter_gradient_history.json', gradients)
            epoch_summary(record, label)
            minimum = cfg.zero_stage_min_epochs if probability == 0 else 1
            if fixed_durations is None and stage_epoch >= minimum and stale >= patience: break
        durations.append(stage_epoch); selected_stage_epochs.append(best_stage_epoch if valid_loader is not None else stage_epoch)
        if valid_loader is not None:
            restored = restore(stage_checkpoint, model, optimizer, device)
            tqdm.write(f'{label}: restored stage {stage+1} epoch {restored["stage_epoch"]}; completed {stage_epoch} epochs')
    selected = (restored if valid_loader is not None else
                checkpoint(model, optimizer, cfg, weights, threshold, epoch, len(durations)-1, selected_stage_epochs[-1]))
    selected['selected_stage_epochs'] = selected_stage_epochs
    selected['epochs_executed'] = epoch
    torch.save(selected, folder / ('best_checkpoint.pt' if valid_loader is not None else 'final_checkpoint.pt'))
    save_history(history, gradients, directories, label)
    write_json(folder / 'fit_summary.json', {'stage_durations': durations, 'selected_stage_epochs': selected_stage_epochs,
                                            'epochs_executed': epoch, 'selection': 'best zero-forcing stage' if valid_loader is not None else 'fixed median fold durations'})
    return durations, history
