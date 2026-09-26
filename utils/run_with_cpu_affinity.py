"""Apply a Windows process CPU mask before running an unchanged Python script."""
from __future__ import annotations

import argparse
import ctypes
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True, type=lambda value: int(value, 0))
    parser.add_argument("script", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script = args.script.resolve(strict=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    process = kernel.GetCurrentProcess()
    current = ctypes.c_size_t()
    available = ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(process, ctypes.byref(current), ctypes.byref(available)):
        raise ctypes.WinError(ctypes.get_last_error())
    if args.mask <= 0 or args.mask & ~current.value:
        raise ValueError(f"Requested mask {args.mask:#x} is outside allowed mask {current.value:#x}")
    if not kernel.SetProcessAffinityMask(process, args.mask):
        raise ctypes.WinError(ctypes.get_last_error())
    applied = ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(process, ctypes.byref(applied), ctypes.byref(available)):
        raise ctypes.WinError(ctypes.get_last_error())
    if applied.value != args.mask:
        raise RuntimeError("CPU affinity verification failed")
    print(f"[CPU-AFFINITY] process mask={applied.value:#x} original={current.value:#x}", flush=True)
    sys.argv = [str(script), *args.arguments]
    sys.path[0] = str(script.parent)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
