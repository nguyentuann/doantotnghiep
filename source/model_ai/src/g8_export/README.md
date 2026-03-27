# G8 — Export ONNX

**File:** `export_onnx.py` (TODO)
**Input:** `models/final/model_best.pt`
**Output:** `models/final/model_best.onnx`

---

## Kế hoạch triển khai

```python
import torch

model = build_model("bilstm")  # hoặc model tốt nhất
model.load_state_dict(torch.load("models/final/model_best.pt"))
model.eval()

dummy_input = torch.randn(1, 8, 12)
torch.onnx.export(
    model,
    dummy_input,
    "models/final/model_best.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17,
)
```

## Verify

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("models/final/model_best.onnx")
x = np.random.randn(1, 8, 12).astype(np.float32)

pt_out  = model(torch.from_numpy(x)).detach().numpy()
ort_out = sess.run(None, {"input": x})[0]

assert np.allclose(pt_out, ort_out, atol=1e-5), "ONNX output không khớp PyTorch!"
print("ONNX verify PASS")
```

## Tại sao cần ONNX?

- Backend FastAPI dùng `onnxruntime` (nhẹ hơn PyTorch ~10×)
- Không cần cài PyTorch trên server
- Inference nhanh hơn cho CPU deployment
