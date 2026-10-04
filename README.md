# Bitcoin Seq2Seq - Experiment 2A / 2B

Research implementation of a **24-hour context (48 candles) to 6-hour forecast (12 candles)** using 30-minute BTC/USDT OHLCV data, a two-layer BiLSTM encoder, Bahdanau attention, and an autoregressive two-layer LSTM decoder.

2A optimizes transformed-candle Smooth-L1 loss. 2B adds volatility-normalized return Smooth-L1 and an auxiliary Bear/Neutral/Bull classification head with cross-entropy. Smoke results verify the pipeline; they do not establish forecasting quality.

## Setup and run

Use Python 3.11 or newer and install the requirements. For CUDA, use a PyTorch installation appropriate to your GPU/driver.

`cudnn_policy = "auto"` conservatively disables cuDNN on Windows with Python 3.14 or newer. On the tested Windows/Python 3.14/PyTorch 2.11 CUDA runtime, cuDNN recurrent training wrote valid outputs but then crashed during interpreter shutdown (`0xC0000409`). Disabling cuDNN keeps CUDA training active and avoids the reproduced crash; it may reduce throughput. Other platforms/older Python versions use cuDNN normally. The resolved backend is recorded in the manifest. Override with `"enabled"` or `"disabled"` in the hyperparameters file if your runtime requires it.

```text
python -m pip install -r requirements.txt
python main.py --experiment 2A --mode validate
python main.py --experiment 2A --mode smoke
python main.py --experiment 2B --mode smoke
python main.py --experiment 2A
python main.py --experiment 2B
python -m pytest -q
```

Edit **hyperparameters.txt** for numerical settings and **paths.txt** for dataset/output paths. Both use TOML syntax. Paths resolve relative to the paths file. A standalone GitHub clone requires you to point `dataset_path` at your local CSV; data, original reports, trained weights, and generated outputs are not committed.

Windows paths must use TOML literal strings (`'D:\folder\dataset.csv'`) or forward-slash basic strings (`"D:/folder/dataset.csv"`). Do not put unescaped backslashes in TOML double-quoted strings. Empty/non-string paths, missing datasets, and output paths beneath an existing file are rejected before a run creates artifacts.

CSV columns must be exactly `datetime,open,high,low,close,volume`, in ascending uninterrupted 30-minute order. Timestamps are preserved as supplied; the dataset does not specify a timezone. The canonical local dataset is `BTC_USDT_30m_Binance_20260913_184821.csv`.

## Mathematics and interfaces

`transforms.py` defines independently callable gap/body/excursion/volume functions and `transform_ohlcv`, `reconstruct_ohlcv`, `log_returns`, `historical_volatility`, `fit_threshold`, `trend_labels`, and six-hour return functions. These operate on PyTorch tensors and preserve gradients.

Features are `[ln(O/previous_C), ln(C/O), ln(H/max(O,C)), ln(min(O,C)/L), ln(1+V)]`. Their first two components sum to the close-to-close logarithmic return. Reconstruction uses the last observed close for step 1 and the previous **predicted** close thereafter. No clipping silently hides invalid reconstructions; nonfinite predictions fail explicitly.

`seq2seq_model.forward(context, targets=None, teacher_forcing=0, generator=None)` returns `predictions [B,12,5]`, `attention [B,12,48]`, and optional `trend_logits [B,12,3]`. Predictions and autoregressive feedback are standardized; teacher-forced targets are standardized ground truth. The model's `physical` method reverses standardization using the active training-fold scaler.

The signed standardized outputs are exactly the raw logits `g_s=a_g`, `b_s=a_b`. For each nonnegative physical channel `x` in `u,d,v`, the standardized output is `l_x + softplus(a_x+c_x)`, where `l_x=-mu_x/s_x` and `c_x=softplus_inverse(mu_x/s_x)`. This is algebraically equivalent to the previous scale-aware physical softplus followed by standardization; it is not ordinary unscaled physical softplus. The patch removes the redundant signed-channel round trip and rounds the lower bound toward the physical interior to prevent tiny negative values at extreme negative logits, under both the original float64 scaler and its float32 model copy. Zero logits map approximately to the transformed training mean. A zero-mean constant channel uses an epsilon-sized softplus offset. No detach/NumPy conversion enters the loss path.

Both heads receive the same shared representation `[decoder output; Bahdanau context]`. The 2B head produces **raw** logits, not probabilities, and is not fed solely by the five regression predictions. Cross-entropy supervises the shared encoder/attention/decoder through this branch. Class order is `0=Bear`, `1=Neutral`, `2=Bull`.

`data.py` validates the dataset, creates five expanding folds in the first 90%, and reserves the final 10% for test. Training stride defaults to 1, evaluation stride to 12. Validation/test use 48 context candles **inside their own partitions** before their first forecast. A preceding close serves only as the first transformation anchor. The first raw dataset row is excluded from transformed samples because its preceding close is unknown.

Scalers and the 33rd-percentile trend threshold are fitted on training data only. The threshold uses flattened absolute normalized future returns from complete training windows (with their configured sample weighting). The origin volatility is population standard deviation of 48 observed returns, frozen across its forecast. Constant feature channels use unit scaling.

Per-step labels compare `z_j=(g_j+b_j)/(sigma_t+epsilon)` to the frozen training threshold `k=Q_0.33(abs(z))`: below `-k` is Bear, above `k` is Bull, and both boundaries belong to Neutral. Six-hour reporting uses `R_6h=sum(r_j)` and `z_6h=R_6h/(sigma_t*sqrt(12)+epsilon)`: **a volatility-scaled cumulative return using the conventional square-root-of-time scaling assumption**. The denominator uses origin-known volatility, not directly observed future six-hour realized volatility.

`training.py` computes `L_2A=Lbase` (Smooth-L1 directly in standardized candle space), or `L_2B=Lbase+lambda_r*Lreturn+lambda_c*Ltrend`. Return Smooth-L1 uses origin-volatility-normalized returns derived differentiably from inverse-standardized `g+b`; trend cross-entropy uses raw auxiliary logits.

Weights are calibrated on the first 32 sequential fold-1 training batches, zero teacher forcing, no optimization, using a separate temporary 2B model. `lambda_r=0.5*m_base/max(m_return,epsilon)` and `lambda_c=0.25*m_base/max(m_trend,epsilon)`, then frozen. The manifest saves means, weights, calibration seed, actual batch/sample counts, and forecast-origin indices for each batch. Smoke mode limits calibration to its 128 training samples (two batches by default); it records the actual count instead of claiming 32 were processed.

Shared CPU initialization is scoped to its own seeded RNG and restores the caller's stream. Classification initialization has a separate scoped seed (`shared seed + head_seed_offset`). DataLoader shuffling owns a generator (`seed + shuffle_seed_offset + fold index`); teacher-forcing Bernoulli choices own a device-specific generator (`seed + teacher_seed_offset`). Calibration uses `seed + calibration_seed_offset`, preserves caller CPU/CUDA RNG, and never reuses actual shuffle or teacher generators. Paired 2A/2B fits consequently start with identical shared parameters, sample order and forcing masks, although learned parameters and forecasts naturally diverge. Shared seeds are `seed + fold index` in CV and `seed` for final fitting; final shuffling omits the fold index. `seed_everything` also seeds stochastic training operations such as dropout. Independent AdamW instances remain unchanged. Validation, test, and calibration always run autoregressively.

## Teacher-forcing stages

Defaults are probabilities `[1,.75,.5,.25,0]`, at most 20 epochs per stage and patience 3. A validation improvement must exceed `min_delta` to reset patience; the strictly lowest observed validation objective selects the best checkpoint even if its improvement is smaller. Patience or the epoch cap ends a stage, restores its best model **and matching optimizer**, and advances to the next probability. The zero stage runs at least five CV epochs; its best checkpoint is selected.

Histories, fit summaries and checkpoints distinguish stage-local, 1-based `stage_best_epoch` from `stage_stop_epoch`. Final development training uses the median of the five **best** epochs per stage, never the patience stopping epochs. With five folds this is an integer; the helper rounds up only for an even-fold verification configuration. The final fit has no validation selection and executes exactly those durations, even if the zero-forcing median is below five. Its best-epoch field is null rather than inventing a validation best. Scaler/threshold are refitted on the full development partition; test windows/features/labels are prepared only after final training finishes. Read-only CSV schema/cadence validation and dataset hashing still inspect the whole file before training.

Smoke mode explicitly reduces training/evaluation samples, exercises the last fold and final fit, and caps each stage at one epoch. Its overrides are recorded in the run manifest and do not edit production configuration.

## Outputs

```text
outputs/2A/                       # likewise 2B
  curves/<run_id>/
  matrices/<run_id>/
  history/<run_id>/
  final reconstruction testing/<run_id>/
```

Run IDs use India time and include the mode. Each epoch has a transient batch bar and a persistent summary line. Histories record losses, metrics, learning rate, teacher forcing, timing, and clipping norms. `pre_clip_global_norm` is explicitly measured, clipping is applied, and `post_clip_global_norm` is explicitly recomputed (the clipping function's return is not used). Epoch norms average batch measurements. Every parameter gets end-of-epoch weight statistics and pooled **pre-clipping** gradient mean/std/min/max, pooled L2 norm, mean batch norm, and missing-gradient status/count. Only small statistic vectors are retained for reporting, not full gradient tensors. Full checkpoint tensors and optimizer states are saved separately.

Reports include both raw and normalized confusion matrices; per-step return/price errors; six-hour direction/trend metrics; return-derived class counts; the separate 2B head metrics; and a persistence baseline (flat price at observed close, unchanged last observed volume). Accuracy is never described as raw price-value forecasting success.

Fold data metadata includes training/validation Bear/Neutral/Bull counts **and proportions**. Final-fit data metadata includes development counts/proportions; held-out data metadata is saved after final evaluation. Smoke records both complete partition distributions and effective sampled distributions. Counts are across forecast target labels (overlapping training windows count repeated target candles). Classification metrics include both actual and predicted proportions. The exact persistence definition is in the manifest and metrics: all predicted O/H/L/C equal the last observed close `C_t`, volume equals the last observed `V_t`, returns are zero and trends are Neutral. The baseline function accepts observed context only.

The last validation block and held-out test have aligned actual/predicted candles, volume panels, close comparisons, timestamped CSVs, and predictions/attention NPZ files. Individual forecast windows keep separate anchors and are drawn as separate lines, including when evaluation stride causes overlaps. Checkpoints and manifests preserve the experiment, scaler, threshold, loss weights, configuration, dataset hash, and package versions.

No full training run is performed merely by installing or importing this project. Full experiments can take substantially longer than smoke checks.

## Comparison scope

The controlled comparison here is **Experiment 2A vs Experiment 2B**, both 48→12. Historical Experiment 1 uses **48→24**. Comparing Experiment 1 to Experiment 2 is therefore not a pure single-variable ablation. Its implementation and archived results remain in the older repository and are not altered by this project or correction patch. No forecasting-performance claims are made until a full research run produces evidence.
