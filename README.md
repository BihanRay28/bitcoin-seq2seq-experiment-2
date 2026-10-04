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

CSV columns must be exactly `datetime,open,high,low,close,volume`, in ascending uninterrupted 30-minute order. Timestamps are preserved as supplied; the dataset does not specify a timezone. The canonical local dataset is `BTC_USDT_30m_Binance_20260913_184821.csv`.

## Mathematics and interfaces

`transforms.py` defines independently callable gap/body/excursion/volume functions and `transform_ohlcv`, `reconstruct_ohlcv`, `log_returns`, `historical_volatility`, `fit_threshold`, `trend_labels`, and six-hour return functions. These operate on PyTorch tensors and preserve gradients.

Features are `[ln(O/previous_C), ln(C/O), ln(H/max(O,C)), ln(min(O,C)/L), ln(1+V)]`. Their first two components sum to the close-to-close logarithmic return. Reconstruction uses the last observed close for step 1 and the previous **predicted** close thereafter. No clipping silently hides invalid reconstructions; nonfinite predictions fail explicitly.

`seq2seq_model.forward(context, targets=None, teacher_forcing=0, generator=None)` returns `predictions [B,12,5]`, `attention [B,12,48]`, and optional `trend_logits [B,12,3]`. Predictions are standardized; the model's `physical` method reverses standardization. Softplus constrains physical excursion and log-volume outputs. A scale-aware softplus offset starts positive outputs near training-feature magnitudes instead of imposing an unrealistic initial candle width.

`data.py` validates the dataset, creates five expanding folds in the first 90%, and reserves the final 10% for test. Training stride defaults to 1, evaluation stride to 12. Validation/test use 48 context candles **inside their own partitions** before their first forecast. A preceding close serves only as the first transformation anchor. The first raw dataset row is excluded from transformed samples because its preceding close is unknown.

Scalers and the 33rd-percentile trend threshold are fitted on training data only. The threshold uses flattened absolute normalized future returns from complete training windows (with their configured sample weighting). The origin volatility is population standard deviation of 48 observed returns, frozen across its forecast. Constant feature channels use unit scaling.

`training.py` computes 2A base loss or 2B base + weighted return + trend losses. Weights are calibrated on fold-1 training only, without optimization, then frozen. 2A and 2B have independent AdamW instances and identical shared initialization; the trend head uses its own initialization RNG. Validation, test, and calibration always run autoregressively.

## Teacher-forcing stages

Defaults are probabilities `[1,.75,.5,.25,0]`, at most 20 epochs per stage and patience 3. A validation improvement must exceed `min_delta`. Patience or the epoch cap ends a stage, restores its best model **and optimizer**, and advances to the next probability. The zero stage runs at least five epochs; its best checkpoint is selected. Final development training uses the ceiling of the median executed duration per stage across the five folds, with no test-driven selection.

Smoke mode explicitly reduces training/evaluation samples, exercises the last fold and final fit, and caps each stage at one epoch. Its overrides are recorded in the run manifest and do not edit production configuration.

## Outputs

```text
outputs/2A/                       # likewise 2B
  curves/<run_id>/
  matrices/<run_id>/
  history/<run_id>/
  final reconstruction testing/<run_id>/
```

Run IDs use India time and include the mode. Each epoch has a transient batch bar and a persistent summary line. Histories record losses, metrics, learning rate, teacher forcing, timing, and clipping norms. Every parameter gets end-of-epoch weight statistics and pooled **pre-clipping** gradient statistics across batches, including missing-gradient counts. Full checkpoint tensors and optimizer states are saved separately.

Reports include both raw and normalized confusion matrices; per-step return/price errors; six-hour direction/trend metrics; return-derived class counts; the separate 2B head metrics; and a persistence baseline (flat price at observed close, unchanged last observed volume). Accuracy is never described as raw price-value forecasting success.

The last validation block and held-out test have aligned actual/predicted candles, volume panels, close comparisons, timestamped CSVs, and predictions/attention NPZ files. Individual forecast windows keep separate anchors and are drawn as separate lines, including when evaluation stride causes overlaps. Checkpoints and manifests preserve the experiment, scaler, threshold, loss weights, configuration, dataset hash, and package versions.

No full training run is performed merely by installing or importing this project. Full experiments can take substantially longer than smoke checks.
