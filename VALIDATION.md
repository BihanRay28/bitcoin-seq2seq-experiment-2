# Implementation validation

Verified on 4 October 2026. This records pipeline verification, not evidence of successful BTC forecasting.

## Automated checks

`python -m pytest -q`: **32 passed** (34.30 seconds on the local verification run).

Coverage includes differentiable transformation/inverse round trips, recursive predicted-close anchors, zero volumes, label boundaries, constant-channel scaling, five expanding folds, sample strides, future-data isolation, configuration validation, attention normalization, shared 2A/2B initialization and CUDA RNG isolation, loss gradients, calibration without optimization, pooled gradient summaries, patience transitions, paired optimizer restoration, minimum zero-forcing epochs, repeated-run preservation, and complete five-fold synthetic pipelines for both experiments.

The synthetic pipeline tests reduce model widths, strides and stage lengths for verification. Production defaults remain in hyperparameters.txt.

## Canonical dataset inspection

`python main.py --experiment 2A --mode validate` passed against 52,560 chronological 30-minute candles. Development contains 47,304 rows; held-out test contains 5,256 rows.

| Fold | Train rows | Train windows | Validation windows |
|---|---:|---:|---:|
| 1 | 23,664 | 23,604 | 390 |
| 2 | 28,392 | 28,332 | 390 |
| 3 | 33,120 | 33,060 | 390 |
| 4 | 37,848 | 37,788 | 390 |
| 5 | 42,576 | 42,516 | 390 |

Final development fit has 47,244 training windows; held-out test has 434 evaluation windows. Counts reflect train stride 1 and evaluation stride 12, with in-partition evaluation context.

## Real-data CUDA smoke checks

Both commands completed with **process exit code 0**:

```text
python main.py --experiment 2A --mode smoke
python main.py --experiment 2B --mode smoke
```

Accepted local run IDs:

- 2A: `20261004_203308_951479_smoke`
- 2B: `20261004_203308_951509_smoke`

Each used the default 128-wide recurrent model, all five teacher-forcing probabilities, 128 training samples, 32 validation/test samples, one epoch per stage, fold 5, and final development retraining. Each saved checkpoints, parameter/gradient histories, configuration and split metadata, calibration where applicable, curves, confusion matrices, and separate last-validation/test reconstruction artifacts.

The runtime was Windows, Python 3.14, PyTorch 2.11.0+cu128, CUDA enabled. The automatic compatibility policy disabled cuDNN. Earlier diagnostic cuDNN runs produced artifacts but crashed at native interpreter shutdown and are not accepted as clean smoke checks. A minimal recurrent-model reproducer and its cuDNN-disabled equivalent isolated this backend behavior; the equivalent exited cleanly.

The held-out reconstruction chart was visually inspected for forecast boundary, 48 observed candles, 12 future candles, aligned timestamp labels, common price scale, and volume panels.

## Remaining research work

The full default-data five-fold training runs have not been executed. Neither experiment has established forecasting quality. Run them separately with the documented full commands, then compare their walk-forward and held-out return/trend metrics against persistence.

Dataset CSVs, original planning reports, model checkpoints, and generated outputs are deliberately outside the source repository.
