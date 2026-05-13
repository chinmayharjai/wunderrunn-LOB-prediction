"""Simple training pipeline for the LOB prediction task using PyTorch.

This script is not part of the baseline submission; it's intended to help you
start experimenting with model architectures and hyperparameters.

Usage (activate the .venv_challenge first):

    python train.py                  # trains on train.parquet, validates on valid.parquet
    python train.py --epochs 5       # change number of epochs
    python train.py --batch-size 4   # smaller/bigger batches
    python train.py --save-model m.pt
    python train.py --export-onnx m.onnx   # export the trained model to ONNX

The training loss uses mean squared error on the two targets, masked by the
``need_prediction`` flag.  After each epoch the script evaluates the model on
the provided validation file using the weighted Pearson metric defined in
utils.py.

Feel free to modify the model class (e.g. add layers, attention, normalisation,
change hidden size) and observe the effect on the validation score.
"""
import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# let scoring utility be reused
from utils import ScorerStepByStep, DataPoint


class LobDataset(Dataset):
    """Dataset returning entire sequences. Loads in chunks to avoid OOM."""

    def __init__(self, parquet_path: str, max_seqs: int = None):
        # read all at once (parquet handles it better than streaming)
        df = pd.read_parquet(parquet_path)
        self.seqs: List[np.ndarray] = []
        self.targs: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        
        for _, g in df.groupby("seq_ix"):
            features = g.iloc[:, 3:35].values.astype(np.float32)
            targets = g.iloc[:, 35:].values.astype(np.float32)
            mask = g["need_prediction"].astype(np.float32).values
            self.seqs.append(features)
            self.targs.append(targets)
            self.masks.append(mask)
            if max_seqs and len(self.seqs) >= max_seqs:
                break

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.targs[idx], self.masks[idx]


def collate_fn(batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
    # convert list of tuples into padded tensors (they're all same length 1000 though)
    xs = np.stack([item[0] for item in batch])
    ys = np.stack([item[1] for item in batch])
    masks = np.stack([item[2] for item in batch])
    # shapes: (B, 1000, 32), (B, 1000, 2), (B, 1000)
    return torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(masks)


class SimpleGRU(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # x: (B, T, input_dim)
        out, _ = self.gru(x)  # out: (B, T, hidden_dim)
        pred = self.head(out)  # (B, T, 2)
        return pred


class SignalAwareModel(nn.Module):
    """Signal-Aware Temporal Model v1: delta features + TCN + GRU + head.

    Returns predictions for each timestep (B, T, 2) to work with the existing
    training loop and masking.
    """

    def __init__(self, input_dim=32, hidden_dim=96):
        super().__init__()
        self.input_dim = input_dim
        self.aug_dim = input_dim * 2

        # Temporal CNN block (Conv1d expects channels-first)
        self.conv1 = nn.Conv1d(self.aug_dim, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)

        # GRU block operates on features produced by convs
        self.gru = nn.GRU(64, hidden_dim, batch_first=True)

        # Normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Output head: produce per-timestep outputs from GRU hidden states
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        # x: (B, T, 32)
        # compute delta features: shift difference, pad first step with zeros
        delta = x[:, 1:, :] - x[:, :-1, :]
        delta = torch.cat([torch.zeros_like(delta[:, :1, :]), delta], dim=1)

        x_aug = torch.cat([x, delta], dim=-1)  # (B, T, 64)

        # Conv1d expects (B, C, T)
        x_c = x_aug.transpose(1, 2)
        x_c = torch.relu(self.conv1(x_c))
        x_c = torch.relu(self.conv2(x_c))
        x_seq = x_c.transpose(1, 2)  # (B, T, channels=64)

        out, _ = self.gru(x_seq)  # (B, T, hidden_dim)

        # apply head per timestep
        out_norm = self.norm(out)
        preds = self.head(out_norm)  # (B, T, 2)
        return preds


class PyTorchPredictionModel:
    """Wrapper that exposes the trained model through the same interface the
    baseline uses, so we can re‑use ScorerStepByStep.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.current_seq = None
        self.history: List[np.ndarray] = []

    def predict(self, data_point: DataPoint):
        # very naive; simply run the model on the entire history each call
        if self.current_seq != data_point.seq_ix:
            self.current_seq = data_point.seq_ix
            self.history = []
        self.history.append(data_point.state.copy())
        if not data_point.need_prediction:
            return None
        inp = torch.tensor(self.history, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = self.model(inp)
        # return last step prediction
        return out[0, -1].numpy()


def evaluate(model: nn.Module, valid_path: str, max_seqs: int = None) -> float:
    ds = LobDataset(valid_path, max_seqs=max_seqs)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
    grader = ScorerStepByStep(valid_path)
    pm = PyTorchPredictionModel(model)
    results = grader.score(pm)
    return results["weighted_pearson"]


def train(
    train_path: str,
    valid_path: str,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
    save_model: str = None,
    export_onnx: str = None,
    train_seqs: int = None,
    valid_seqs: int = None,
):
    ds = LobDataset(train_path, max_seqs=train_seqs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # choose architecture
    # default: SignalAwareModel which returns (B,T,2)
    model = SignalAwareModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def weighted_pearson_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, T, 2); mask: (B, T)
        pred_clamped = torch.clamp(pred, -6.0, 6.0)
        target_clamped = torch.clamp(target, -6.0, 6.0)

        losses = []
        eps = 1e-6
        mask = mask.float()
        for i in range(pred_clamped.shape[-1]):
            x = pred_clamped[:, :, i]
            y = target_clamped[:, :, i]
            w = torch.abs(y) * mask
            w_sum = w.sum()
            if w_sum.item() == 0:
                losses.append(torch.tensor(1.0, device=pred.device))
                continue
            mean_x = (w * x).sum() / (w_sum + eps)
            mean_y = (w * y).sum() / (w_sum + eps)
            vx = x - mean_x
            vy = y - mean_y
            cov = (w * vx * vy).sum() / (w_sum + eps)
            var_x = (w * vx * vx).sum() / (w_sum + eps)
            var_y = (w * vy * vy).sum() / (w_sum + eps)
            corr = cov / (torch.sqrt(var_x + eps) * torch.sqrt(var_y + eps) + eps)
            losses.append(1.0 - corr)
        loss = torch.stack(losses).mean()
        return loss

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            pred = model(x)  # (B,T,2)
            loss = weighted_pearson_loss(pred, y, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
        avg_loss = total_loss / count
        print(f"Epoch {epoch}/{epochs}  training loss={avg_loss:.6f}")
        if valid_path:
            model.eval()
            val_score = evaluate(model, valid_path, max_seqs=valid_seqs)
            print(f"          validation weighted pearson: {val_score:.6f}")
    if save_model:
        torch.save(model.state_dict(), save_model)
        print("Saved model weights to", save_model)
    if export_onnx:
        dummy = torch.randn(1, 1000, 32, device=device)
        torch.onnx.export(
            model,
            dummy,
            export_onnx,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
        )
        print("Exported ONNX model to", export_onnx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="datasets/train.parquet")
    parser.add_argument("--valid", default="datasets/valid.parquet")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-model", help="path to store .pt weights")
    parser.add_argument("--export-onnx", help="path to write ONNX file")
    parser.add_argument("--train-seqs", type=int, default=None, help="limit training sequences (for quick iteration)")
    parser.add_argument("--valid-seqs", type=int, default=None, help="limit validation sequences (for quick iteration)")
    args = parser.parse_args()
    train(
        args.train,
        args.valid,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_model=args.save_model,
        export_onnx=args.export_onnx,
        train_seqs=args.train_seqs,
        valid_seqs=args.valid_seqs,
    )
