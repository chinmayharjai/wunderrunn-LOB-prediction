# Minimal stub of onnxscript to satisfy import during torch.onnx.export
# This does NOT implement full onnxscript functionality; it's a light shim

# Provide a simple placeholder object to avoid ModuleNotFoundError
class _Stub:
    pass

# export a few common names if referenced
onnxscript = _Stub()

__all__ = ["onnxscript"]
