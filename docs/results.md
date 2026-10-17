# Results so far

All numbers are on CardRF: 10 consumer drones, one anechoic chamber, one receiver, 1 s (2.4 GHz) or 0.5 s (5.8 GHz) per recording. A model sees one 65,536-sample window (0.55 ms at 120 MS/s) as a 64 x 64 spectrogram. Three seeds change only the training sample and initialisation, not the data, so the spread between seeds is **not** an error bar: windows from one recording are strongly correlated, and real uncertainty is larger than the seed spread suggests. Treat differences of a few points as unestablished.

## 1. Closed-set baseline (`scripts/run_closed_set.py`)

| Test | Accuracy (mean of 3 seeds) |
| --- | --- |
| Validation blocks (65-75% of each recording) | 78.6% |
| **Test blocks** (80-100% of each recording, 1,140 windows) | **77.2%** (77.3, 77.0, 77.3) |
| **Cross-recording**: train on part 1, test on part 2, four two-file drones | **68.1%** (67.1, 67.1, 70.0) |

Cross-recording, per held-out file: DJI Phantom 4 Pro Plus 5G 85.0%, DJI Matrice 210 5G 80.5%, DJI Inspire 2 5G 67.8%, Yuneec Typhoon H 2G 38.9%. The two parts of a recording are not consecutive stretches of one session, so this is the nearest the dataset gets to a new recording. It still comes from the same chamber and receiver.

The task is not trivial: 77% is far from the near-100% that a random window split would probably give, which is why the split holds out time blocks (not measured here, to avoid claiming a number I did not run).

## 2. Where the errors are (`scripts/diagnose_windows.py`, `scripts/run_gated.py`)

Per-drone accuracy ranges from 59% (Typhoon H) to 98% (Phantom 4). The cause is mostly **empty windows**: frequency-hopping links are silent most of the time, so a 0.55 ms window often holds only noise. With an energy detector (a window passes if at least 30% of its 256-sample frames are 6 dB above the recording's own noise floor; both numbers fixed before looking at results):

| Drone | Windows passing the detector | Accuracy on detected windows (model trained on all windows) | (model trained only on detected windows) |
| --- | --- | --- | --- |
| DJI Phantom 4 | 88% | 100% | 100% |
| DJI Mavic Pro | 95% | 94.7% | 98.4% |
| DJI Matrice 100 | 89% | 98.8% | 97.2% |
| DJI Matrice 210 | 87% | 88.8% | 94.1% |
| DJI Inspire 2 | 88% | 82.6% | 86.6% |
| DJI Phantom 4 Pro Plus | 88% | 84.5% | 86.9% |
| DJI Mavic Mini | 54% | 100% | 100% |
| Parrot Disco | 13% | 100% | 100% |
| Parrot Mambo | 10% | 75% | 100% |
| Yuneec Typhoon H | 14% | 99.3% | 100% |

(Mean of 3 seeds. For the three drones with a 10-14% pass rate, only a few dozen test windows are detected, so their 100% means "few windows, all right", not a precise rate.)

Overall, 61.7% of test windows pass the gate (4,026 of 6,527). On those, the ungated model scores **89.6%** and a model trained only on detected windows **92.8%**. As a share of *all* test windows, "detected and correct" is 55.3% for the ungated model and 57.2% for the gated one.

What this says:

- **The 77% headline mostly measures how often a window contains a burst.** When a burst is present, classification is 93% or better for 7 of the 10 drones, and 100% for the sparse hoppers that are caught.
- **The real weaknesses are two.** (1) Detection coverage for sparse hoppers (Parrot Disco and Mambo, Yuneec Typhoon H): a fixed 0.55 ms window catches a burst only 10-14% of the time, so a deployed system would have to accumulate or use longer windows. (2) Confusion between DJI siblings: Inspire 2, Phantom 4 Pro Plus and Matrice 210 stay at 83-94% even on busy windows. I have not yet looked at a confusion matrix, so I do not know which pairs are confused or why.
- **Gated training helps a little and consistently** (about 3 points on detected windows, same sign in all three seeds), but it is one dataset and one run per seed.
- **The noise floor used by the gate comes from the whole recording.** A live receiver would estimate it from past samples only. Not tested.

## Not done yet

Longer windows or burst-level classification for the sparse hoppers; a confusion matrix; the simulated-shift sweeps (SNR, frequency offset, I/Q imbalance, multipath, narrow or low-bit receivers) and training with augmentation; open-set detection; the DroneRF dataset (verified and being converted); any real-time or on-device measurement.
