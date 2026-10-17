# SpectrumSentry

Receive-only classification of drone radio links from raw IQ samples, with an honest look at how it breaks when conditions change.

**Status: early. A closed-set baseline on CardRF is measured (77% on held-out time blocks, 68% on a held-out recording, 93% on windows that contain a burst; see [results](docs/results.md)). The simulated-shift experiments, open-set detection and the second dataset are not done.** Nothing here transmits; it analyses recordings. See [design](docs/design.md) for scope and for what the data cannot show.

## What it will do

1. Classify the drone model from a half-millisecond window of IQ (a spectrogram CNN), with splits that hold out whole time blocks and, where possible, whole recordings.
2. Measure how accuracy falls under simulated noise, frequency offset, I/Q imbalance, multipath and a narrower or cheaper receiver, and how much training on those impairments recovers.
3. Flag drones the model has never seen (open-set), by holding out drone models.

## Data

[CardRF](https://zenodo.org/records/4264467) (Tampere University, CC BY 4.0): 10 consumer drones, raw IQ, recorded in an anechoic chamber. About 8.6 GB. Download with `python scripts/download_cardrf.py` (resumable, checksum-verified). The data is not committed.

A second dataset, [DroneRF](https://data.mendeley.com/datasets/f4c2b4n755/1) (Al-Sa'd, Allahham, Mohamed, Al-Ali, Khattab and Erbad, Qatar University, 2019; Mendeley Data V1, DOI 10.17632/f4c2b4n755.1; CC BY 4.0), is planned as an independent test: a different team, place and receivers, with three drones (AR, Bebop, Phantom) plus background RF. It is downloaded by hand from the Mendeley page and its SHA-256 is checked before use. It is not committed either.

## Run

```sh
pip install -e ".[dev]"
pytest
python scripts/run_closed_set.py --seeds 3
```

## Licence

MIT. The dataset has its own licence (CC BY 4.0, attribution to its authors).
