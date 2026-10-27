
import os
import sys
import json
from pathlib import Path
from collections import Counter

import numpy as np
import onnxruntime as ort

print("=== ONNX Runtime ===")
print("Version:", ort.__version__)
print("Available providers:", ort.get_available_providers())

# Keep all registered DLL directories alive through inference.
site_packages = Path(sys.prefix) / "Lib" / "site-packages"
nvidia_dir = site_packages / "nvidia"
dll_handles = []

print("\n=== Register NVIDIA DLL directories ===")

if nvidia_dir.exists():
    for dll_dir in nvidia_dir.rglob("bin"):
        if dll_dir.is_dir():
            try:
                dll_handles.append(os.add_dll_directory(str(dll_dir)))
                print("Registered:", dll_dir)
            except OSError as e:
                print("Registration failed:", dll_dir, repr(e))
else:
    print("NVIDIA package directory not found:", nvidia_dir)

print("\n=== Preload CUDA/cuDNN DLLs ===")
try:
    ort.preload_dlls(directory="")
    print("ONNX Runtime DLL preload: SUCCESS")
except Exception as e:
    print("ONNX Runtime DLL preload: FAILED")
    print(type(e).__name__, repr(e))
    raise

model_path = (
    r"C:\Users\jeet_\.insightface\models"
    r"\buffalo_l\w600k_r50.onnx"
)

profile_prefix = str(Path.cwd() / "gpu_test_profile")

print("\n=== Create ONNX session ===")
session = ort.InferenceSession(
    model_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    sess_options=ort.SessionOptions()
)

print("Session providers:", session.get_providers())

# Enable profiling on a separate session so we can inspect execution.
options = ort.SessionOptions()
options.enable_profiling = True
options.profile_file_prefix = profile_prefix

profile_session = ort.InferenceSession(
    model_path,
    sess_options=options,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

print("\n=== Actual model inference ===")
model_input = profile_session.get_inputs()[0]
print("Input:", model_input.name, model_input.shape)

dummy_input = np.zeros((1, 3, 112, 112), dtype=np.float32)

outputs = profile_session.run(
    None,
    {model_input.name: dummy_input}
)

print("Inference completed.")
print("Output shapes:", [output.shape for output in outputs])
print("Active session providers:", profile_session.get_providers())

print("\n=== Inspect execution profile ===")
profile_path = profile_session.end_profiling()
print("Profile file:", profile_path)

with open(profile_path, "r", encoding="utf-8") as f:
    events = json.load(f)

provider_counts = Counter()

for event in events:
    args = event.get("args", {})
    provider = args.get("provider")

    if provider:
        provider_counts[provider] += 1

print("Execution-provider event counts:", dict(provider_counts))

if provider_counts.get("CUDAExecutionProvider", 0) > 0:
    print("RESULT: CUDA execution events detected.")
elif provider_counts.get("CPUExecutionProvider", 0) > 0:
    print("RESULT: CPU execution events detected; CUDA not confirmed.")
else:
    print("RESULT: No provider-tagged events found; inspect the profile.")

print("\n=== Test complete ===")
