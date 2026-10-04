"""Terminal entrypoint: python main.py --experiment 2A|2B."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import sys
from types import SimpleNamespace
import tomllib

import torch
from torch.utils.data import Subset

from data import (CandleWindows, chronological_split, class_distribution, describe_split,
                  final_datasets, fold_datasets, load_dataset, loader)
from model import seq2seq_model
from output import PERSISTENCE_DEFINITION, aggregate_folds, create_run, confusion_charts, save_forecasts, write_json
from training import calibrate_temporary, evaluate, final_stage_durations, fit, seed_everything

HERE = Path(__file__).resolve().parent


def load_configuration(hyperparameters=HERE / 'hyperparameters.txt', paths=HERE / 'paths.txt'):
    hyperparameters, paths = Path(hyperparameters), Path(paths)
    sections = tomllib.loads(hyperparameters.read_text(encoding='utf-8'))
    settings = {}
    for name, section in sections.items():
        if not isinstance(section, dict): raise ValueError('Use named TOML sections for hyperparameters')
        for key, value in section.items():
            if key in settings: raise ValueError(f'Duplicate hyperparameter {key}')
            settings[key] = value
    expected = {'context_length', 'horizon', 'input_size', 'candle_minutes', 'train_stride', 'eval_stride',
                'test_fraction', 'n_folds', 'threshold_quantile', 'epsilon', 'volatility_ddof',
                'encoder_hidden_size', 'decoder_hidden_size', 'num_layers', 'attention_dim', 'head_hidden_size',
                'dropout', 'learning_rate', 'weight_decay', 'batch_size', 'gradient_clip', 'smooth_l1_beta',
                'seed', 'head_seed_offset', 'teacher_seed_offset', 'shuffle_seed_offset', 'calibration_seed_offset',
                'device', 'num_workers', 'pin_memory',
                'cpu_threads', 'cudnn_policy', 'teacher_probabilities', 'stage_max_epochs', 'stage_patience',
                'zero_stage_min_epochs', 'min_delta', 'calibration_batches', 'return_contribution',
                'trend_contribution', 'smoke_train_samples', 'smoke_eval_samples', 'smoke_stage_epochs'}
    if set(settings) != expected:
        raise ValueError(f'Configuration missing={sorted(expected-set(settings))}, unknown={sorted(set(settings)-expected)}')
    cfg = SimpleNamespace(**settings)
    validate_configuration(cfg)
    path_values = tomllib.loads(paths.read_text(encoding='utf-8'))
    if set(path_values) != {'dataset_path', 'output_root'}:
        raise ValueError('paths.txt needs exactly dataset_path and output_root')
    if any(not isinstance(v, str) or not v.strip() for v in path_values.values()):
        raise ValueError('Configuration paths must be nonempty TOML strings')
    resolved = {key: (paths.parent / value).resolve() for key, value in path_values.items()}
    return cfg, resolved


def validate_paths(paths):
    if not Path(paths['dataset_path']).is_file():
        raise ValueError(f'Dataset path must be an existing CSV file: {paths["dataset_path"]}')
    root = Path(paths['output_root'])
    for candidate in (root, *root.parents):
        if candidate.exists():
            if not candidate.is_dir(): raise ValueError('Output path must resolve under a directory, not a file')
            break


def validate_configuration(cfg):
    if (cfg.context_length, cfg.horizon, cfg.input_size, cfg.candle_minutes, cfg.num_layers) != (48, 12, 5, 30, 2):
        raise ValueError('Final contract requires 48 input, 12 output, 5 features, 30-minute candles, two layers')
    positive_integers = ['train_stride', 'eval_stride', 'n_folds', 'encoder_hidden_size', 'decoder_hidden_size',
                         'attention_dim', 'head_hidden_size', 'batch_size', 'cpu_threads', 'zero_stage_min_epochs',
                         'calibration_batches', 'smoke_train_samples', 'smoke_eval_samples', 'smoke_stage_epochs']
    for key in positive_integers:
        value = getattr(cfg, key)
        if type(value) is not int or value <= 0: raise ValueError(f'{key} must be a positive integer')
    for key in ['seed', 'head_seed_offset', 'teacher_seed_offset', 'shuffle_seed_offset', 'calibration_seed_offset', 'num_workers']:
        if type(getattr(cfg, key)) is not int or getattr(cfg, key) < 0: raise ValueError(f'{key} must be nonnegative integer')
    for key in ['epsilon', 'learning_rate', 'gradient_clip', 'smooth_l1_beta', 'return_contribution', 'trend_contribution']:
        value = getattr(cfg, key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ['weight_decay', 'min_delta']:
        value = getattr(cfg, key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if not 0 < cfg.test_fraction < .5 or not 0 < cfg.threshold_quantile < 1:
        raise ValueError('test_fraction must be (0,.5) and threshold_quantile (0,1)')
    if not 0 <= cfg.dropout < 1 or cfg.volatility_ddof != 0 or type(cfg.pin_memory) is not bool:
        raise ValueError('Invalid dropout, volatility_ddof (must be 0), or pin_memory')
    if cfg.device not in ('auto', 'cpu', 'cuda'): raise ValueError('device must be auto, cpu, or cuda')
    if cfg.cudnn_policy not in ('auto', 'enabled', 'disabled'):
        raise ValueError('cudnn_policy must be auto, enabled, or disabled')
    probabilities = cfg.teacher_probabilities
    if not probabilities or probabilities[0] != 1 or probabilities[-1] != 0 or any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Teacher probabilities must start at 1 and end at 0')
    if any(a <= b for a, b in zip(probabilities, probabilities[1:])):
        raise ValueError('Teacher probabilities must strictly decrease')
    for key in ['stage_max_epochs', 'stage_patience']:
        values = getattr(cfg, key)
        if len(values) != len(probabilities) or any(type(v) is not int or v <= 0 for v in values):
            raise ValueError(f'{key} must give one positive integer per stage')
    if cfg.stage_max_epochs[-1] < cfg.zero_stage_min_epochs:
        raise ValueError('Zero-stage cap must cover its required minimum epochs')


def select_device(cfg):
    if cfg.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    return torch.device('cuda' if cfg.device != 'cpu' and torch.cuda.is_available() else 'cpu')


def configure_backend(cfg, device):
    compatibility = platform.system() == 'Windows' and sys.version_info >= (3, 14)
    enabled = cfg.cudnn_policy == 'enabled' or (cfg.cudnn_policy == 'auto' and not compatibility)
    torch.backends.cudnn.enabled = enabled
    if device.type == 'cuda' and not enabled:
        print('cuDNN disabled by runtime compatibility policy; training still uses CUDA.')
    return {'cudnn_enabled': enabled, 'cudnn_policy': cfg.cudnn_policy,
            'python_version': platform.python_version(), 'platform': platform.platform()}


def limited(dataset, count, last=False):
    start = max(0, len(dataset)-count) if last else 0
    return Subset(dataset, range(start, min(start+count, len(dataset))))


def run(experiment, mode, cfg, paths):
    validate_paths(paths)
    frame = load_dataset(paths['dataset_path'], cfg)
    split = describe_split(frame, cfg)
    print(json.dumps({'experiment': experiment, 'mode': mode, 'configuration': vars(cfg),
                      'paths': {k: str(v) for k,v in paths.items()}, 'split': split}, indent=2))
    if mode == 'validate':
        dev_end, folds = chronological_split(len(frame), cfg)
        counts = []
        for fold in folds:
            train, valid = fold_datasets(frame, fold, cfg)
            counts.append({'fold': fold.index, 'train_samples': len(train), 'validation_samples': len(valid),
                           'threshold': train.threshold, 'scaler': train.scaler.to_dict(),
                           'training_classes': class_distribution(train), 'validation_classes': class_distribution(valid)})
        final_train, test = final_datasets(frame, dev_end, cfg)
        print(json.dumps({'folds': counts, 'final_train_samples': len(final_train), 'test_samples': len(test),
                          'development_classes': class_distribution(final_train), 'held_out_classes': class_distribution(test)}, indent=2))
        return None
    torch.set_num_threads(cfg.cpu_threads)
    device = select_device(cfg)
    backend = configure_backend(cfg, device)
    run_id, directories = create_run(paths['output_root'], experiment, mode)
    manifest_path = directories['history'] / 'run.json'
    metadata = {'run_id': run_id, 'experiment': experiment, 'mode': mode, 'status': 'running',
                'configuration': vars(cfg).copy(), 'paths': {k:str(v) for k,v in paths.items()}, 'split': split,
                'device': str(device), 'backend': backend,
                'persistence_definition': PERSISTENCE_DEFINITION,
                'six_hour_normalization': 'a volatility-scaled cumulative return using the conventional square-root-of-time scaling assumption',
                'rng_streams': {'shared_initialization': 'seed + fold index; final fit uses seed',
                                'classification_initialization': 'shared initialization seed + head_seed_offset',
                                'shuffle': 'seed + shuffle_seed_offset + fold index; final fit omits fold index',
                                'teacher_forcing': cfg.seed + cfg.teacher_seed_offset,
                                'calibration': cfg.seed + cfg.calibration_seed_offset},
                'dataset_sha256': hashlib.sha256(paths['dataset_path'].read_bytes()).hexdigest(),
                'package_versions': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'pandas', 'matplotlib', 'tqdm')},
                'outputs': {k:str(v) for k,v in directories.items()}}
    write_json(manifest_path, metadata)
    try:
        dev_end, folds = chronological_split(len(frame), cfg)
        # Calibration always uses fold-1 training, even when smoke exercises the final fold.
        if experiment == '2B':
            calibration_train = CandleWindows(frame, 0, folds[0].train_end, cfg, cfg.train_stride)
            calibration_data = limited(calibration_train, cfg.smoke_train_samples) if mode == 'smoke' else calibration_train
            weights = calibrate_temporary(calibration_train.scaler, loader(calibration_data, cfg), cfg, device)
            del calibration_train, calibration_data
        else:
            weights = {'return': 0., 'trend': 0., 'calibration': None}
        metadata['loss_weights'] = weights; write_json(manifest_path, metadata)
        print(f'Device={device}; loss weights: return={weights["return"]:.6g}, trend={weights["trend"]:.6g}')
        if mode == 'smoke':
            # Explicit, recorded test-only overrides; production settings remain unchanged.
            cfg = SimpleNamespace(**vars(cfg))
            cfg.stage_max_epochs = [cfg.smoke_stage_epochs] * len(cfg.teacher_probabilities)
            cfg.zero_stage_min_epochs = cfg.smoke_stage_epochs
            metadata['effective_configuration'] = vars(cfg).copy()
            metadata['smoke_limits'] = {'fold_indices': [folds[-1].index], 'train_samples': cfg.smoke_train_samples,
                                        'evaluation_samples': cfg.smoke_eval_samples}
        durations = []; fold_metrics = []
        for fold in (folds[-1:] if mode == 'smoke' else folds):
            train, valid = fold_datasets(frame, fold, cfg)
            train_part = limited(train, cfg.smoke_train_samples) if mode == 'smoke' else train
            valid_part = limited(valid, cfg.smoke_eval_samples, last=True) if mode == 'smoke' else valid
            label = f'fold_{fold.index:02d}'
            seed_everything(cfg.seed + fold.index)
            model = seq2seq_model(cfg, train.scaler, experiment, initialization_seed=cfg.seed + fold.index).to(device)
            shuffle_seed = cfg.seed + cfg.shuffle_seed_offset + fold.index
            train_loader, valid_loader = loader(train_part, cfg, True, shuffle_seed), loader(valid_part, cfg)
            print(f'{label}: train_samples={len(train_part)} validation_samples={len(valid_part)} threshold={train.threshold:.6g}')
            write_json(directories['history']/label/'data.json', {'fold': asdict(fold), 'scaler': train.scaler.to_dict(),
                                                                 'threshold': train.threshold, 'train_samples':len(train_part),
                                                                 'validation_samples':len(valid_part),
                                                                 'training_classes':class_distribution(train),
                                                                 'validation_classes':class_distribution(valid),
                                                                 'effective_training_classes':class_distribution(train_part),
                                                                 'effective_validation_classes':class_distribution(valid_part),
                                                                 'rng_seeds':{'shared': model.initialization_seed,
                                                                              'classification': model.initialization_seed + cfg.head_seed_offset,
                                                                              'shuffle':shuffle_seed,
                                                                              'teacher_forcing':cfg.seed + cfg.teacher_seed_offset}})
            best_epochs, _ = fit(model, train_loader, valid_loader, cfg, weights, device, train.threshold, directories, label)
            durations.append(best_epochs)
            metrics, arrays = evaluate(model, valid_loader, cfg, weights, device, train.threshold)
            fold_metrics.append({'fold': fold.index, 'metrics': metrics})
            write_json(directories['history']/label/'metrics.json', metrics)
            confusion_charts(metrics, directories, label)
            if fold.index == folds[-1].index:
                save_forecasts(arrays, metrics, frame, directories, 'last_validation', cfg)
        final_train = CandleWindows(frame, 0, dev_end, cfg, cfg.train_stride)
        fixed = final_stage_durations(durations, len(cfg.teacher_probabilities), 1 if mode == 'smoke' else cfg.n_folds)
        final_part = limited(final_train, cfg.smoke_train_samples) if mode == 'smoke' else final_train
        seed_everything(cfg.seed)
        model = seq2seq_model(cfg, final_train.scaler, experiment).to(device)
        write_json(directories['history']/'final_fit'/'data.json', {'scaler':final_train.scaler.to_dict(),
                   'threshold':final_train.threshold, 'stage_durations':fixed, 'train_samples':len(final_part),
                   'schedule_source':'median CV stage_best_epoch (stage-local, 1-based)', 'fold_stage_best_epochs':durations,
                   'development_classes':class_distribution(final_train),
                   'effective_development_classes':class_distribution(final_part)})
        fit(model, loader(final_part, cfg, True), None, cfg, weights, device, final_train.threshold, directories, 'final_fit', fixed)
        # No test features, labels, windows, or outcome statistics are constructed
        # until final fitting is complete. Dataset schema validation is read-only.
        test = CandleWindows(frame, dev_end, len(frame), cfg, cfg.eval_stride, final_train.scaler, final_train.threshold)
        test_part = limited(test, cfg.smoke_eval_samples, last=True) if mode == 'smoke' else test
        metrics, arrays = evaluate(model, loader(test_part, cfg), cfg, weights, device, final_train.threshold)
        write_json(directories['history']/'final_fit'/'held_out_data.json',
                   {'test_samples':len(test_part), 'held_out_classes':class_distribution(test),
                    'effective_held_out_classes':class_distribution(test_part), 'threshold':final_train.threshold,
                    'constructed_after_final_fit':True})
        save_forecasts(arrays, metrics, frame, directories, 'held_out_test', cfg)
        confusion_charts(metrics, directories, 'held_out_test')
        aggregate = aggregate_folds(fold_metrics, directories)
        write_json(directories['history']/'metrics.json', {'folds':fold_metrics, 'walk_forward_aggregate':aggregate, 'held_out_test':metrics})
        metadata.update({'status':'complete', 'final_stage_durations':fixed, 'folds_completed':len(fold_metrics),
                         'final_schedule_source':'median CV stage_best_epoch', 'fold_stage_best_epochs':durations,
                         'result_kind':'pipeline verification only' if mode == 'smoke' else 'full experiment'})
        write_json(manifest_path, metadata)
        print(f'Completed {experiment} {mode}: {directories["history"]}')
        return directories
    except Exception as error:
        metadata.update({'status':'failed', 'error':str(error)}); write_json(manifest_path, metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description='48 observed BTC candles -> 12 future candles')
    parser.add_argument('--experiment', required=True, choices=['2A','2B'])
    parser.add_argument('--mode', choices=['full','smoke','validate'], default='full')
    parser.add_argument('--hyperparameters', type=Path, default=HERE/'hyperparameters.txt')
    parser.add_argument('--paths', type=Path, default=HERE/'paths.txt')
    args = parser.parse_args()
    cfg, paths = load_configuration(args.hyperparameters, args.paths)
    run(args.experiment, args.mode, cfg, paths)


if __name__ == '__main__':
    main()
