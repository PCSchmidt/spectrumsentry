"""Convert the DroneRF CSV segments (inside RAR archives) to compact int16 .npy files.

    python scripts/convert_dronerf.py

Input: data/dronerf/raw/DroneRF/**/*.rar (unzipped from the Mendeley download, SHA-256 verified).
Each RAR holds one-row CSVs of ~10 million real-valued samples (one 0.25 s segment of one half,
"L" or "H", of the 2.4 GHz band). Windows' bsdtar (libarchive) reads the RARs. Each CSV is
extracted to a scratch folder, parsed, stored as int16 in data/dronerf/npy/<code><L|H>_<segment>.npy
and deleted, so the 44 GB of text never accumulates. Existing outputs are skipped.

Class code (from the dataset paper, consistent with the folder names): five digits
  d t t m m   d = drone present; tt = 00 Bebop, 01 AR, 10 Phantom; mm = flight mode
  (00 on and connected, 01 hovering, 10 flying, 11 video recording). 00000 = background.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "dronerf" / "raw" / "DroneRF"
OUT = ROOT / "data" / "dronerf" / "npy"
SCRATCH = ROOT / "data" / "dronerf" / "scratch"
TAR = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / "tar.exe"
_NAME = re.compile(r"(\d{5})([LH])_(\d+)\.csv$")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for rar in sorted(RAW.rglob("*.rar")):
        shutil.rmtree(SCRATCH, ignore_errors=True)
        SCRATCH.mkdir(parents=True)
        listing = subprocess.run([str(TAR), "-tf", str(rar)], capture_output=True, text=True, check=True).stdout.splitlines()
        members = [m for m in listing if _NAME.search(m)]
        todo = [m for m in members if not (OUT / (lambda g: f"{g[0]}{g[1]}_{g[2]}.npy")(_NAME.search(m).groups())).exists()]
        print(f"{rar.name}: {len(members)} segments, {len(todo)} to convert", flush=True)
        if not todo:
            continue
        # Extract the whole archive once: pulling single members out of a solid RAR re-decompresses
        # everything before them, which made per-member extraction take hours.
        subprocess.run([str(TAR), "-xf", str(rar), "-C", str(SCRATCH)], check=True)
        for m in todo:
            csv = SCRATCH / m
            code, half, seg = _NAME.search(m).groups()
            # split + array is ~17x faster than np.fromstring(sep=",") on these one-row CSVs and gives identical values
            x = np.array(csv.read_bytes().split(b","), dtype=np.float64)
            if not np.array_equal(x, np.round(x)) or np.abs(x).max() > 32767:
                raise SystemExit(f"{m}: values are not int16-representable integers")
            np.save(OUT / f"{code}{half}_{seg}.npy", x.astype(np.int16))
            csv.unlink()
    shutil.rmtree(SCRATCH, ignore_errors=True)
    n = len(list(OUT.glob("*.npy")))
    print(f"done: {n} segment files")


if __name__ == "__main__":
    main()
