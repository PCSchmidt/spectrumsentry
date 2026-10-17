# SpectrumSentry

Receive-only classification of drone radio links from raw IQ samples, with an honest look at how it breaks when conditions change.

**Status: early. A closed-set baseline on CardRF is measured (77% on held-out time blocks, 68% on a held-out recording, 93% on windows that contain a burst; see [results](docs/results.md)). Simulated shift is measured too: a clean-trained model falls from 97% to 21% with 10 dB of added noise, and training on impairments recovers most of it but not receiver bandwidth. A second dataset (DroneRF) shows the noise fragility does not repeat there (its model holds to about 5 dB) while flight-mode recognition suffers from drift over the recording session. Open-set: distance in the CNN's embedding space separates unknown drones well (AUROC 0.94) where softmax confidence does not (0.70, below chance for DJI siblings), though at a 95%-known threshold it still accepts 37% of unknown windows on average.** Nothing here transmits; it analyses recordings. See [design](docs/design.md) for scope and for what the data cannot show.

## What it will do

1. Classify the drone model from a half-millisecond window of IQ (a spectrogram CNN), with splits that hold out whole time blocks and, where possible, whole recordings.
2. Measure how accuracy falls under simulated noise, frequency offset, I/Q imbalance, multipath and a narrower or cheaper receiver, and how much training on those impairments recovers.
3. Flag drones the model has never seen (open-set), by holding out drone models. Done: see results.

## Data

[CardRF](https://zenodo.org/records/4264467) (Tampere University, CC BY 4.0): 10 consumer drones, raw IQ, recorded in an anechoic chamber. About 8.6 GB. Download with `python scripts/download_cardrf.py` (resumable, checksum-verified). The data is not committed.

A second dataset, [DroneRF](https://data.mendeley.com/datasets/f4c2b4n755/1) (Al-Sa'd, Allahham, Mohamed, Al-Ali, Khattab and Erbad, Qatar University, 2019; Mendeley Data V1, DOI 10.17632/f4c2b4n755.1; CC BY 4.0), is used as an independent check: a different team, place and receivers, with three drones (AR, Bebop, Phantom) plus background RF. It is downloaded by hand from the Mendeley page (3.75 GB zip), its SHA-256 is checked, and `scripts/convert_dronerf.py` turns its CSV segments into compact arrays. It is not committed either.

## Run

```sh
pip install -e ".[dev]"
pytest
python scripts/run_closed_set.py --seeds 3
```

## Licence

MIT. The dataset has its own licence (CC BY 4.0, attribution to its authors).
