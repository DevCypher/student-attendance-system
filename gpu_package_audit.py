
import sys
import os
import importlib.metadata as metadata
from pathlib import Path

print("=== Python environment ===")
print("Executable:", sys.executable)
print("Prefix:", sys.prefix)

print("\n=== Relevant installed packages ===")

keywords = (
    "onnxruntime",
    "nvidia-cuda",
    "nvidia-cudnn",
    "nvidia-cublas",
    "nvidia-cufft",
    "nvidia-curand",
    "nvidia-nvjitlink",
    "nvidia-cuda-nvrtc",
)

packages = []

for dist in metadata.distributions():
    name = dist.metadata.get("Name", "")
    normalized = name.lower().replace("_", "-")

    if any(key in normalized for key in keywords):
        packages.append((name, dist.version))

for name, version in sorted(set(packages)):
    print(f"{name} == {version}")

print("\n=== NVIDIA package DLL locations ===")

site_packages = Path(sys.prefix) / "Lib" / "site-packages"
nvidia_dir = site_packages / "nvidia"

if nvidia_dir.exists():
    for dll_name in (
        "cudnn64_9.dll",
        "cudnn_engines_tensor_ir64_9.dll",
        "cudart64_12.dll",
        "cublas64_12.dll",
    ):
        matches = list(nvidia_dir.rglob(dll_name))
        print(f"\n{dll_name}:")
        if matches:
            for path in matches:
                print(" ", path)
        else:
            print("  Not found under this environment's nvidia directory")
else:
    print("NVIDIA directory not found:", nvidia_dir)

print("\n=== CUDA/cuDNN-related PATH entries ===")

for entry in os.environ.get("PATH", "").split(os.pathsep):
    low = entry.lower()
    if any(term in low for term in ("cuda", "cudnn", "nvidia")):
        print(entry)

print("\n=== ONNX Runtime debug information ===")

try:
    import onnxruntime as ort
    print("ONNX Runtime version:", ort.__version__)
    print("Available providers:", ort.get_available_providers())
    ort.print_debug_info()
except Exception as e:
    print("Debug information failed:", repr(e))

print("\n=== Audit complete ===")
