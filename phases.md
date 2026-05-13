# Wunder Challenge 2 — Learning & Improvement Journey

## Goal
Learn deeply from this LOB prediction project, analyze every part like a data scientist
with LOB knowledge, then find and implement a better model architecture.

---

## Phase Status

| Phase | Title | Status |
|-------|-------|--------|
| 1 | Full Repo Walkthrough (File by File) | ✅ Done |
| 2 | Data Analysis (EDA like a quant/LOB expert) | ⬜ Next |
| 3 | Architecture Research & Selection | ⬜ Pending |
| 4 | Implementation of New Model | ⬜ Pending |
| 5 | Training, Evaluation & Iteration | ⬜ Pending |

---

## Phase 1: Full Repo Walkthrough ✅

### Problem Summary
- **Competition:** Wunder Challenge 2 — LOB Price Movement Prediction
- **Task:** Predict two future price movement indicators (t0, t1) from LOB + trade data
- **Data:** 10,721 train sequences / 1,444 valid sequences, each exactly 1000 steps
- **Scored region:** Steps 99–999 (901 steps per sequence; first 99 = warm-up)
- **Metric:** Weighted Pearson Correlation — weights by |y_true| so large moves matter more
- **Submission format:** `solution.py` with `PredictionModel` class, `predict(data_point)` method

### Features (32 total)
| Group | Columns | Description |
|-------|---------|-------------|
| Bid Prices | p0–p5 | 6 levels of bid side LOB prices |
| Ask Prices | p6–p11 | 6 levels of ask side LOB prices |
| Bid Volumes | v0–v5 | 6 levels of bid side volumes |
| Ask Volumes | v6–v11 | 6 levels of ask side volumes |
| Trade Prices | dp0–dp3 | Recent trade price features |
| Trade Volumes | dv0–dv3 | Recent trade volume features |

### Targets
- `t0`, `t1` — two anonymized future price movement indicators

### File Map
```
competition_package/
├── README.md                  — Problem spec & submission format
├── utils.py                   — DataPoint, ScorerStepByStep, weighted_pearson_correlation
├── train.py                   — PyTorch training pipeline (SimpleGRU + SignalAwareModel)
├── export_untrained.py        — ONNX export utility
├── datasets/
│   ├── train.parquet          — 10,721 sequences (~1.5GB)
│   └── valid.parquet          — 1,444 sequences (~225MB)
└── example_solution/
    ├── solution.py            — Vanilla GRU baseline (ONNX inference)
    ├── baseline.onnx          — Pre-trained baseline weights
    └── small_valid.parquet    — Quick test dataset

Submission_1/
├── solution.py                — SignalAwareModel (PyTorch) + ONNX fallback
└── signal_aware_v1.pt         — Trained PyTorch checkpoint

Submission_2/
└── solution.py                — Standard ONNX baseline (same as example)

Submission_3/
└── solution.py                — Duplicate of Submission_1

evaluate_submissions.py        — Batch evaluator for all 3 submissions
analysis.ipynb                 — EDA notebook (data exploration)
```

### Key Classes & Functions

#### `utils.py`
- `DataPoint` — dataclass: `(seq_ix, step_in_seq, need_prediction, state: np.ndarray[32])`
- `weighted_pearson_correlation(y_true, y_pred)` — clips preds to [-6,6], weights by |y_true|
- `ScorerStepByStep(dataset_path)` — loads parquet, iterates row-by-row, calls model.predict()
- `ScorerStepByStep.score(model)` — returns `{t0: float, t1: float, weighted_pearson: float}`

#### `train.py`
- `LobDataset` — PyTorch Dataset, groups by seq_ix, returns (features[1000,32], targets[1000,2], mask[1000])
- `collate_fn` — stacks sequences into batches: (B,1000,32), (B,1000,2), (B,1000)
- `SimpleGRU` — GRU(32→64) + Linear(64→2), window=100 at inference
- `SignalAwareModel` — delta features → Conv1d×2 → GRU(64→96) → LayerNorm → MLP(96→64→2)
- `weighted_pearson_loss` — directly optimizes evaluation metric (1 - weighted_pearson)
- `PyTorchPredictionModel` — wraps nn.Module to expose predict() interface
- `train()` — full training loop with Adam optimizer, per-epoch validation

#### `example_solution/solution.py`
- `PredictionModel` — loads baseline.onnx via onnxruntime, uses last 100 steps as window
- State reset on new seq_ix
- Run: `python solution.py` or `python solution.py small_valid.parquet`

#### `Submission_1/solution.py`
- `PredictionModel` — loads signal_aware_v1.pt (PyTorch) → falls back to ONNX
- Same SignalAwareModel architecture as train.py
- Same 100-step window at inference

#### `evaluate_submissions.py`
- Dynamically imports each submission via `importlib.util.spec_from_file_location`
- Scores all 3 submissions, prints comparison table
- Run: `python evaluate_submissions.py`

### Model Architecture Evolution
| Version | Architecture | Notes |
|---------|-------------|-------|
| Baseline | SimpleGRU: GRU(32→64) + Linear | Raw features, window=100 |
| Sub_1 | SignalAwareModel: delta→Conv1d×2→GRU(64→96)→LN→MLP | Delta features, local patterns |

### Known Limitations (to improve in later phases)
1. **100-step window** at inference — wastes the 900+ steps of available history
2. **No normalization** of input features — raw LOB values may have scale issues
3. **Simple delta features** — more sophisticated feature engineering possible
4. **GRU only** — no attention mechanism to focus on important historical moments
5. **No cross-sequence patterns** — each sequence treated independently from cold start

---

## Phase 2: Data Analysis (EDA) ⬜ — NEXT

### Goals for Phase 2
A data scientist + LOB expert would ask:

1. **Target distribution** — are t0/t1 symmetric? fat-tailed? How many zeros?
2. **Target autocorrelation** — how many steps ahead are targets predictive?
3. **Feature distributions** — are prices normalized? What scale are volumes on?
4. **Feature-target correlations** — which features are most predictive of t0/t1?
5. **Order book imbalance** — (bid_vol - ask_vol)/(bid_vol + ask_vol) — classic LOB signal
6. **Spread** — (best_ask - best_bid) — volatility proxy
7. **Temporal patterns** — does predictability change over the 1000-step window?
8. **Large move analysis** — what market conditions precede large |t0|/|t1|?
9. **t0 vs t1 correlation** — are they the same signal at different horizons?

### Commands to run analysis
```bash
# activate venv first
.venv/Scripts/activate   # Windows

# open the existing analysis notebook
jupyter notebook analysis.ipynb

# or run quick pandas analysis
python -c "
import pandas as pd
import numpy as np
df = pd.read_parquet('competition_package/datasets/train.parquet')
print(df.shape)
print(df.describe())
"
```

---

## Phase 3: Architecture Research & Selection ⬜

### Candidate architectures to evaluate (after data analysis informs the choice)
- **Transformer with causal attention** — attends to full 1000-step history
- **Temporal Convolutional Network (TCN)** — dilated convolutions for multi-scale patterns
- **Mamba / S4** — state space models, efficient for long sequences
- **Hybrid: CNN + LSTM/GRU with attention** — local + global features
- **Ensemble** — combine multiple model outputs

### LOB-specific features to engineer
- Order book imbalance at each level
- Weighted mid-price (volume-weighted)
- Spread (best_ask - best_bid)
- Volume at touch vs. depth ratios
- Trade flow imbalance (buy vs. sell volume)
- Rolling statistics (mean, std, min, max over windows)

---

## Phase 4: Implementation ⬜

_(To be detailed after Phase 3)_

---

## Phase 5: Training & Evaluation ⬜

_(To be detailed after Phase 4)_

---

## Notes & Observations
_(Running log — add findings as we go)_

- Sequences are independent and randomly shuffled
- Warm-up period (steps 0-98) exists to let the model build context
- Both targets weighted equally in final score
- Predictions clipped to [-6, 6] — no benefit to predicting extreme values
- The competition data is fully anonymized — no asset name, exchange, or timestamps
