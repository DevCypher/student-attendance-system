
import onnxruntime as ort

# Load NVIDIA CUDA/cuDNN libraries before initializing InsightFace
ort.preload_dlls(directory="")

from insightface.app import FaceAnalysis

print("Available providers:", ort.get_available_providers())

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(320, 320))

print("\nModel execution providers:")

for name, model in app.models.items():
    print(f"{name}: {model.session.get_providers()}")

print("\nGPU diagnostic complete.")
