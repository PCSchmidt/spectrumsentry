"""Download the CardRF drone IQ recordings (Zenodo, CC BY 4.0) with size and MD5 verification.

    python scripts/download_cardrf.py

Source: "Radio-Frequency Control and Video Signal Recordings of Drones",
Zenodo record 4264467, DOI 10.5281/zenodo.4264467 (Tampere University). 25 files, about 8.6 GB.
Resumable (HTTP Range) and retried, because this network drops DNS intermittently. Files that
already match their recorded size and MD5 are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "cardrf"
RECORD = "https://zenodo.org/api/records/4264467"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, size: int, attempts: int = 40) -> None:
    for attempt in range(1, attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have == size:
            return
        if have > size:
            dest.unlink()
            have = 0
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=60) as resp, dest.open("ab" if have else "wb") as f:
                while block := resp.read(1 << 20):
                    f.write(block)
        except OSError as e:
            print(f"  attempt {attempt} for {dest.name} failed at {dest.stat().st_size if dest.exists() else 0}/{size}: {e}", flush=True)
            time.sleep(min(30, 3 * attempt))
    raise SystemExit(f"giving up on {dest.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0, help="this worker's index when files are split across parallel workers")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(urllib.request.urlopen(RECORD, timeout=60).read())
    if args.worker == 0:
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    for i, f in enumerate(sorted(manifest["files"], key=lambda f: f["key"])):
        if i % args.workers != args.worker:
            continue
        dest = OUT / f["key"]
        want = f["checksum"].split(":", 1)[1]
        if dest.exists() and dest.stat().st_size == f["size"] and md5(dest) == want:
            print(f"ok (cached) {f['key']}", flush=True)
            continue
        print(f"downloading {f['key']} ({f['size'] / 1e6:.0f} MB)", flush=True)
        fetch(f["links"]["self"], dest, f["size"])
        if md5(dest) != want:
            dest.unlink()
            raise SystemExit(f"checksum mismatch for {f['key']}; deleted, rerun")
        print(f"ok {f['key']}", flush=True)
    print("all files verified")


if __name__ == "__main__":
    sys.exit(main())
