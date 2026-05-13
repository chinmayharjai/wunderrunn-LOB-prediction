# Minimal stub of onnxscript to satisfy import during torch.onnx.export
# Placed at competition_package so torch can import it when running export_untrained.py

class _Stub:
    pass

onnxscript = _Stub()

__all__ = ["onnxscript"]
