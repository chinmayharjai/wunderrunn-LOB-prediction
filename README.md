# Wunder Challenge 2 — LOB Price Movement Prediction

A machine learning competition project for predicting future price movements from **Limit Order Book (LOB)** data — a sequence modeling problem from the domain of high-frequency trading.

## Problem

Given a stream of market states (LOB + trade data), predict two future price movement indicators `t0` and `t1` at each timestep.

- **10,721 training sequences**, each 1000 timesteps long
- **32 features** per timestep: bid/ask prices (6 levels each), bid/ask volumes (6 levels each), trade prices and volumes
- **Metric:** Weighted Pearson Correlation — emphasizes large price movements

## Model Evolution

| Submission | Architecture | Key Idea |
|---|---|---|
| Baseline | Vanilla GRU (hidden=64) | Raw features, ONNX inference |
| Submission 1 & 3 | SignalAwareModel | Delta features + Conv1d×2 + GRU (hidden=96) + LayerNorm + MLP head |

## Architecture: SignalAwareModel

```
Input (B, T, 32)
  → Delta features: x[t] - x[t-1]       # rate-of-change signals
  → Concatenate [x, delta] → (B, T, 64)
  → Conv1d × 2 (kernel=5)               # local temporal patterns
  → GRU (hidden=96)                     # sequence memory
  → LayerNorm → MLP (96→64→2)
Output (B, T, 2)                        # per-timestep predictions
```

## Training

```bash
cd wnn_predictorium_starterpack/competition_package

# activate venv
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt

# train
python train.py --epochs 10 --batch-size 8 --save-model my_model.pt

# evaluate baseline
cd example_solution && python solution.py

# evaluate all submissions
cd .. && python evaluate_submissions.py
```

## Key Files

```
competition_package/
├── utils.py          # DataPoint, ScorerStepByStep, weighted_pearson_correlation
├── train.py          # Training pipeline: SimpleGRU + SignalAwareModel
├── README.md         # Competition spec
└── example_solution/
    └── solution.py   # Vanilla GRU baseline (ONNX)

Submission_1/
└── solution.py       # SignalAwareModel with PyTorch checkpoint

evaluate_submissions.py   # Batch evaluator for all submissions
analysis.ipynb            # EDA: distributions, autocorrelation, LOB signals
phases.md                 # Learning journey & improvement roadmap
```

## Loss Function

Directly optimizes the evaluation metric:
```python
loss = 1 - weighted_pearson_correlation(pred, target, weights=|target|)
```

## Data

Dataset not included (large files). The competition provides:
- `datasets/train.parquet` (~1.5 GB, 10,721 sequences)
- `datasets/valid.parquet` (~215 MB, 1,444 sequences)

## Roadmap

- [x] Baseline GRU (ONNX)
- [x] SignalAwareModel (delta features + TCN + GRU)
- [ ] Deep EDA — autocorrelation, LOB imbalance, large-move analysis
- [ ] Transformer / Mamba architecture
- [ ] Advanced feature engineering (OBI, spread, VWAP)
- [ ] Ensemble strategy
