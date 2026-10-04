# Implementation validation

Verified on 4 October 2026. This records pipeline verification, not evidence of successful BTC forecasting.

## Automated checks

Correction verification: `python -m pytest -q`: **46 passed, 0 failed** (29.76 seconds on the local verification run). The initial implementation had 32 tests; all existing valid coverage is retained and extended.

Coverage includes differentiable transformation/inverse round trips, recursive predicted-close anchors, zero volumes, label boundaries, constant-channel scaling, five expanding folds, sample strides, future-data isolation, configuration validation, attention normalization, shared 2A/2B initialization and CUDA RNG isolation, loss gradients, calibration without optimization, pooled gradient summaries, patience transitions, paired optimizer restoration, minimum zero-forcing epochs, repeated-run preservation, and complete five-fold synthetic pipelines for both experiments.

Added coverage checks extreme negative constrained logits under both float32 model scaling and the original float64 training scaler; zero logits near training means; classification and regression heads receiving the identical shared tensor; CE gradients reaching shared modules; independent head seeds; calibration preserving CPU/CUDA, actual shuffle and teacher RNG; paired sample order and actual decoder Bernoulli masks; stage BEST versus STOP epochs (including tiny improvements that do not reset patience); median-of-five BEST scheduling; independently measured clipping norms; gradient missing-status/norm fields; final-development statistics unaffected by held-out mutations; observed-only persistence; delayed test-window construction; class proportions; and literal/forward-slash TOML Windows paths.

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

- 2A: `20261004_205417_423815_smoke`
- 2B: `20261004_205428_536744_smoke`

Each used the default 128-wide recurrent model, all five teacher-forcing probabilities, 128 training samples, 32 validation/test samples, one epoch per stage, fold 5, and final development retraining. Each saved checkpoints, parameter/gradient histories, configuration and split metadata, calibration where applicable, curves, confusion matrices, and separate last-validation/test reconstruction artifacts.

Post-run artifact checks passed for both: 69 files for 2A and 78 for 2B. Saved physical predictions were `[32,12,5]`, attention `[32,12,48]` with normalized rows, and 2B raw trend logits `[32,12,3]`. All epoch/evaluation losses and gradient summaries were finite. Physical excursions/log-volume were nonnegative; reconstructed prices were positive, volumes nonnegative, highs above open/close and lows below open/close. All clipping post-norms respected the configured limit. Both schedules used the smoke fold's BEST epochs `[1,1,1,1,1]`; this is a pipeline check, not a substitute for five-fold production selection.

2B smoke calibration recorded seed 40042, 128 samples, two actual sequential fold-1 batches and their origin indices. Full mode retains the configured first-32-batch calibration contract. Read-only configuration/data validation also passed with the canonical dataset and counts shown above.

The runtime was Windows, Python 3.14, PyTorch 2.11.0+cu128, CUDA enabled. The automatic compatibility policy disabled cuDNN. Earlier diagnostic cuDNN runs produced artifacts but crashed at native interpreter shutdown and are not accepted as clean smoke checks. A minimal recurrent-model reproducer and its cuDNN-disabled equivalent isolated this backend behavior; the equivalent exited cleanly.

The initial implementation's reconstruction chart was visually inspected for forecast boundary, 48 observed candles, 12 future candles, aligned timestamp labels, common price scale, and volume panels. The correction preserves the plotting code; new-run chart existence, tensor geometry and OHLCV validity were rechecked programmatically.

## Correction audit

Only the new `BihanRay28/bitcoin-seq2seq-experiment-2` repository is patched. The older Experiment 1 repository is untouched and its local checkout remains clean.

| Changed files | Correction |
|---|---|
| `model.py` | Explicit seeded shared initialization; independent head initialization; exact signed standardized channels; rounding-safe nonnegative lower bounds. |
| `training.py` | Separate stage BEST/STOP records; paired restoration retained; final schedule helper uses BEST epochs; isolated temporary calibration with audit fields; explicit pre/post clipping measurements. |
| `main.py` | Use corrected schedule/calibration; seed metadata; path validation; class distributions; postpone held-out window preparation until after final fitting. |
| `data.py` | Independent shuffle seed offset and target-label distribution helper; fold/scaler/threshold algorithms unchanged. |
| `output.py` | Preserve baseline values but expose observed-only helper/exact definition; add proportions and gradient norm/missing status. |
| `hyperparameters.txt` | Add shuffle/calibration seed offsets; scientific defaults unchanged. |
| Existing three test files | Extend constraint, branch, RNG, schedule, clipping, causality, path and artifact checks. |
| `README.md`, `VALIDATION.md` | Document corrected behavior, verification provenance and comparison limits. |

Inspected and retained: transformation formulas and recursive anchors, 48→12 forecast contract, BiLSTM/LSTM/Bahdanau architecture, classification shared-representation branch, differentiable base/return/CE objectives, training-only threshold/scaler fitting, origin-causal volatility and square-root-of-time six-hour normalization, persistence predictions, plotting and cuDNN compatibility policy. `transforms.py`, `paths.txt`, requirements, CI workflow and `.gitignore` need no edits.

Histories include both stage-local BEST and STOP epochs. CV zero-forcing still runs at least five epochs; the final fit executes exactly the median BEST duration, which may be shorter. Strictly lowest objective selects the checkpoint; `min_delta` controls patience resets separately. These are different concepts and are tested independently.

The extra `positive_lower` checkpoint buffer means the new strict model state schema differs from initial smoke checkpoints. Old artifacts are preserved but are not resume inputs to this entrypoint (which has no resume option). Independent seeds do not promise bitwise equality across different platforms, hardware or PyTorch versions; versions/backend are recorded in each run manifest.

## Remaining research work

The full default-data five-fold training runs have not been executed. Neither experiment has established forecasting quality. Run them separately with the documented full commands, then compare their walk-forward and held-out return/trend metrics against persistence.

Dataset CSVs, original planning reports, model checkpoints, and generated outputs are deliberately outside the source repository.
