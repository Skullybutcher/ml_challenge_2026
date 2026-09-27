#!/usr/bin/env python3
"""
Launch parallel inference after E5 retrain completes.
Run this after E5 generation completes and retrain finishes.
"""
import subprocess
import sys
from pathlib import Path

def main():
    DATA_DIR = Path(r"D:\amlc\student_resource\dataset")
    MODEL_DIR = Path(r"D:\amlc\repo2\ml_challenge_2026\exp_run1\out_full")  # Adjust if different
    OUT_DIR = Path(r"D:\amlc\repo2\ml_challenge_2026\exp_run1\out_full_final")
    E5_NAME = Path(r"C:\mlc_model\e5_name")
    E5_ADDR = Path(r"E:\mlc_model\e5_address")
    
    cmd = [
        sys.executable, "parallel_inference.py",
        "--data-dir", str(DATA_DIR),
        "--out-dir", str(OUT_DIR),
        "--model-dir", str(MODEL_DIR),
        "--workers", "8",
        "--test-chunk-size", "25000",
        "--use-e5",
        "--e5-name-dir", str(E5_NAME),
        "--e5-address-dir", str(E5_ADDR),
        "--france-threshold", "0.95",
        "--other-threshold", "0.65",
        "--use-rare",
        "--rare-max-df", "1000"
    ]
    
    print("Launching parallel inference...")
    print("Command:", " ".join(cmd))
    
    result = subprocess.run(cmd, cwd=r"D:\amlc\repo2\ml_challenge_2026")
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())