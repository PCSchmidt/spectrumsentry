"""Figure for docs/results.md: macro accuracy under each simulated impairment, three training regimes.

    python scripts/make_figure.py

Reads results/shift.json. Lines are the mean over seeds, bands the min-max over seeds. The dashed line
is chance (10 drones). Colours are the Okabe-Ito set, chosen to be distinguishable under common colour
vision deficiencies; lines also differ in marker so the legend does not rely on colour alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STYLE = {
    "clean": ("#0072B2", "o", "trained clean"),
    "aug_all": ("#D55E00", "s", "trained with all impairments"),
    "aug_no_rx": ("#009E73", "^", "trained without receiver impairments"),
}
TITLES = {
    "snr_db": "Added noise (SNR, dB)",
    "cfo_max_hz": "Carrier frequency offset (max, Hz)",
    "iq_imbalance_gain_db/phase_deg": "I/Q imbalance (dB / degrees)",
    "multipath_decay/spacing": "Multipath (decay / spacing in samples)",
    "receiver_bandwidth_mhz": "Receiver bandwidth (MHz)",
    "adc_bits": "ADC resolution (bits)",
}


def main() -> None:
    d = json.loads((ROOT / "results" / "shift.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (sweep, levels) in zip(axes.flat, d["sweeps"].items()):
        labels = list(levels)
        x = np.arange(len(labels))
        for m, (colour, marker, name) in STYLE.items():
            vals = np.array([levels[l][m] for l in labels]) * 100
            ax.plot(x, vals.mean(1), color=colour, marker=marker, lw=2, ms=6, label=name)
            ax.fill_between(x, vals.min(1), vals.max(1), color=colour, alpha=0.15, lw=0)
        ax.axhline(10, color="#555555", ls="--", lw=1)
        ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(TITLES[sweep], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("macro accuracy (%)")
    axes[1, 0].set_ylabel("macro accuracy (%)")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("SpectrumSentry: simulated shift on CardRF (detected test windows; dashed = chance)", fontsize=12)
    out = ROOT / "docs" / "shift.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
