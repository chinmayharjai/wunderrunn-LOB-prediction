"""Quick export of untrained SignalAwareModel to ONNX for testing."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SignalAwareModel(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=96):
        super().__init__()
        self.input_dim = input_dim
        self.aug_dim = input_dim * 2
        self.conv1 = nn.Conv1d(self.aug_dim, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.gru = nn.GRU(64, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        delta = x[:, 1:, :] - x[:, :-1, :]
        delta = torch.cat([torch.zeros_like(delta[:, :1, :]), delta], dim=1)
        x_aug = torch.cat([x, delta], dim=-1)
        x_c = x_aug.transpose(1, 2)
        x_c = torch.relu(self.conv1(x_c))
        x_c = torch.relu(self.conv2(x_c))
        x_seq = x_c.transpose(1, 2)
        out, _ = self.gru(x_seq)
        out_norm = self.norm(out)
        preds = self.head(out_norm)
        return preds

# Create and export
model = SignalAwareModel()
dummy = torch.randn(1, 1000, 32)
torch.onnx.export(
    model, dummy, "signal_aware_untrained.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
)
print("Exported signal_aware_untrained.onnx")
