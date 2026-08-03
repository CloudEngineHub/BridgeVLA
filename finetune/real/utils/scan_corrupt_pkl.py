"""Scan all .pkl files under the dataset root and report corrupted ones.

Usage:
    python scan_corrupt_pkl.py [data_root]

Default data_root:
    $REAL_DATA_ROOT (exported by real/train.sh), else
    <repo>/data/bridgevla_data/Real derived from this file's location.
"""

import os
import pickle
import sys
from pathlib import Path


def scan(data_root: str):
    data_root = Path(data_root)
    if not data_root.is_dir():
        print(f"ERROR: {data_root} is not a directory")
        sys.exit(1)

    total = 0
    corrupt = []

    pkl_files = sorted(data_root.rglob("*.pkl"))
    print(f"Found {len(pkl_files)} .pkl files under {data_root}")

    for p in pkl_files:
        total += 1
        try:
            with open(p, "rb") as f:
                pickle.load(f)
        except Exception as e:
            corrupt.append((str(p), type(e).__name__, str(e)))
            print(f"  CORRUPT: {p}  ({type(e).__name__}: {e})")

    print(f"\nScanned {total} files, found {len(corrupt)} corrupt file(s).")
    if corrupt:
        print("\nCorrupt file list:")
        for path, etype, msg in corrupt:
            print(f"  {path}")
    else:
        print("All files are OK.")


if __name__ == "__main__":
    # This file lives at <BRIDGEVLA_ROOT>/finetune/real/utils/, so parents[3]
    # is the repo root. REAL_DATA_ROOT (exported by real/train.sh) wins
    # when present.
    default_root = os.environ.get("REAL_DATA_ROOT") or str(
        Path(__file__).resolve().parents[3] / "data" / "bridgevla_data" / "Real")
    root = sys.argv[1] if len(sys.argv) > 1 else default_root
    scan(root)
