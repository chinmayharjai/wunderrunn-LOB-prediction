import os
import sys
import numpy as np

# Adjust path to import utils from parent directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")

from utils import DataPoint, ScorerStepByStep
# try to use PyTorch model if available, otherwise fall back to ONNX
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None

import onnxruntime as ort

class PredictionModel:
    """Submission model: preferentially load a PyTorch checkpoint `signal_aware_v1.pt`.

    If the checkpoint is not present or `torch` is unavailable, falls back to the
    baseline ONNX model (`baseline.onnx`).
    """

    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.sequence_history = []

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.torch_model = None

        # Attempt to load torch checkpoint
        ckpt_path = os.path.join(base_dir, "signal_aware_v1.pt")
        if torch is not None and os.path.exists(ckpt_path):
            try:
                # Define minimal SignalAwareModel matching training architecture
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

                model = SignalAwareModel()
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                model.eval()
                self.torch_model = model
                print(f"Loaded PyTorch checkpoint from {ckpt_path}")
            except Exception as e:
                print(f"Error loading PyTorch model: {e}")
                self.torch_model = None

        # Prepare ONNX fallback
        onnx_path = os.path.join(base_dir, "baseline.onnx")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session = None
        try:
            if os.path.exists(onnx_path):
                self.ort_session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
                self.input_name = self.ort_session.get_inputs()[0].name
                self.output_name = self.ort_session.get_outputs()[0].name
                print(f"Loaded ONNX model from {onnx_path}")
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            self.ort_session = None

    def predict(self, data_point: DataPoint) -> np.ndarray:
        # Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        # Update history
        self.sequence_history.append(data_point.state.copy())

        # If prediction not needed yet, return None
        if not data_point.need_prediction:
            return None

        # If torch model available, use it
        if self.torch_model is not None:
            # prepare last 100 steps
            history_window = self.sequence_history[-100:]
            if len(history_window) == 0:
                return np.zeros(2)
            if len(history_window) < 100:
                padding = [np.zeros_like(history_window[0])] * (100 - len(history_window))
                history_window = padding + history_window
            data_arr = np.asarray(history_window, dtype=np.float32)
            data_tensor = torch.from_numpy(np.expand_dims(data_arr, axis=0))
            with torch.no_grad():
                out = self.torch_model(data_tensor)
            # out: (1, T, 2)
            pred = out[0, -1].cpu().numpy()
            return pred

        # Otherwise fall back to ONNX
        if self.ort_session is None:
            return np.zeros(2)

        history_window = self.sequence_history[-100:]
        if len(history_window) < 100:
             padding = [np.zeros_like(history_window[0])] * (100 - len(history_window))
             history_window = padding + history_window

        data_arr = np.asarray(history_window, dtype=np.float32)
        data_tensor = np.expand_dims(data_arr, axis=0)
        ort_inputs = {self.input_name: data_tensor}
        output = self.ort_session.run([self.output_name], ort_inputs)[0]
        if len(output.shape) == 3:
            prediction = output[0, -1, :]
        else:
            prediction = output[0]
        return prediction


if __name__ == "__main__":
    # Local testing
    # allow overriding the parquet file via command-line argument (e.g. small_valid.parquet)
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = f"{CURRENT_DIR}/../datasets/valid.parquet"
    
    if os.path.exists(test_file):
        model = PredictionModel()
        scorer = ScorerStepByStep(test_file)
        
        print("Testing Vanilla GRU Baseline (ONNX)...")
        results = scorer.score(model)
        
        print("\nResults:")
        print(f"Mean Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
        for i, target in enumerate(scorer.targets):
            print(f"  {target}: {results[target]:.6f}")
    else:
        print("Valid parquet not found for testing.")
