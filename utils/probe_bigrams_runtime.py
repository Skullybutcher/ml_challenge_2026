"""Bounded pure-Python probe for the observed comprehension runtime failures."""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
import traceback


def original_bigrams(s: str) -> set:
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else set()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--affinity-mask", type=lambda s: int(s, 0))
    args = parser.parse_args()
    if args.affinity_mask is not None:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), args.affinity_mask):
            raise ctypes.WinError(ctypes.get_last_error())

    cases = ["", "a", "a b", "north south service company limited",
             "rue fran\u00e7aise soci\u00e9t\u00e9 num\u00e9ro 123", "abcdefghijklmnopqrstuvwxyz0123456789"]
    expected = [set(map("".join, zip(s, s[1:]))) for s in cases]
    print(f"probe python={sys.version!r} pid={os.getpid()} affinity={args.affinity_mask}", flush=True)
    started = time.monotonic()
    next_report = started + 10
    iterations = 0
    try:
        while time.monotonic() - started < args.seconds:
            for _ in range(1000):
                for s, target in zip(cases, expected):
                    result = original_bigrams(s)
                    if result != target:
                        raise AssertionError(f"Incorrect bigrams at iteration={iterations}, input_type={type(s)!r}")
                    iterations += 1
            if time.monotonic() >= next_report:
                print(f"progress elapsed={time.monotonic()-started:.1f}s checks={iterations}", flush=True)
                next_report += 10
    except Exception:
        print(f"FAIL checks={iterations} elapsed={time.monotonic()-started:.1f}s", flush=True)
        traceback.print_exc()
        return 1
    print(f"PASS checks={iterations} elapsed={time.monotonic()-started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
