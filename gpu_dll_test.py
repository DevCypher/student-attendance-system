
import os
import sys
import ctypes
from pathlib import Path

print("=== Python environment ===")
print("Python executable:", sys.executable)
print("Python version:", sys.version)

cudnn_dir = (
    Path(sys.prefix)
    / "Lib"
    / "site-packages"
    / "nvidia"
    / "cudnn"
    / "bin"
)

target_dll = cudnn_dir / "cudnn_engines_tensor_ir64_9.dll"

print("\n=== DLL diagnostics ===")
print("cuDNN directory:", cudnn_dir)
print("Directory exists:", cudnn_dir.is_dir())
print("Target DLL:", target_dll)
print("Target DLL exists:", target_dll.is_file())

dll_directory_handle = None

try:
    dll_directory_handle = os.add_dll_directory(str(cudnn_dir))
    print("DLL directory registration: SUCCESS")
except OSError as e:
    print("DLL directory registration: FAILED")
    print(repr(e))

print("\n=== Loading Tensor IR DLL ===")

try:
    dll = ctypes.WinDLL(str(target_dll))
    print("DLL LOAD: SUCCESS")
    print("Library handle:", dll._handle)
except OSError as e:
    print("DLL LOAD: FAILED")
    print("Exception:", repr(e))
    print("Windows error code:", getattr(e, "winerror", None))
    print("Windows error message:", getattr(e, "strerror", None))

print("\n=== Diagnostic complete ===")

if dll_directory_handle is not None:
    dll_directory_handle.close()
