"""Which recording is which: parse CardRF file names into drone model, band and part."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Sampling rate (Hz) and length (samples) per band, from the dataset description.
BANDS = {"2G": {"fs": 120e6, "centre": 2.44e9, "n": 120_000_000}, "5G": {"fs": 200e6, "centre": 5.8e9, "n": 100_000_000}}
_NAME = re.compile(r"^(?P<drone>.+?)_(?:(?P<link>control|video)_)?(?P<band>2G|5G)(?:_(?P<part>\d)of(?P<parts>\d))?\.bin$")


@dataclass(frozen=True)
class Recording:
    path: Path
    drone: str  # the class label, e.g. "DJI_inspire_2" (a drone's control and video files share it)
    band: str  # "2G" or "5G"
    link: str  # "control", "video" or "" (the file holds both / unspecified)
    part: int  # 1 or 2 for drones recorded in two separate files, else 0

    @property
    def fs(self) -> float:
        return BANDS[self.band]["fs"]

    @property
    def n_samples(self) -> int:
        return BANDS[self.band]["n"]


def parse(path: Path) -> Recording:
    m = _NAME.match(path.name)
    if not m:
        raise ValueError(f"not a CardRF recording name: {path.name}")
    return Recording(path, m["drone"], m["band"], m["link"] or "", int(m["part"] or 0))


def catalog(data_dir: Path) -> list[Recording]:
    return sorted((parse(p) for p in data_dir.glob("*.bin")), key=lambda r: (r.drone, r.band, r.part, r.link))
