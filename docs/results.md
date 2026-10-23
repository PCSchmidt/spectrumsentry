# Results so far

All numbers are on CardRF: 10 consumer drones, one anechoic chamber, one receiver, 1 s (2.4 GHz) or 0.5 s (5.8 GHz) per recording. A model sees one 65,536-sample window (0.55 ms at 120 MS/s) as a 64 x 64 spectrogram. Three seeds change only the training sample and initialisation, not the data, so the spread between seeds is **not** an error bar: windows from one recording are strongly correlated, and real uncertainty is larger than the seed spread suggests. Treat differences of a few points as unestablished.

## 1. Closed-set baseline (`scripts/run_closed_set.py`)

| Test | Accuracy (mean of 3 seeds) |
| --- | --- |
| Validation blocks (65-75% of each recording) | 78.6% |
| **Test blocks** (80-100% of each recording, 1,140 windows) | **77.2%** (77.3, 77.0, 77.3) |
| **Cross-recording**: train on part 1, test on part 2, four two-file drones | **68.1%** (67.1, 67.1, 70.0) |

Cross-recording, per held-out file: DJI Phantom 4 Pro Plus 5G 85.0%, DJI Matrice 210 5G 80.5%, DJI Inspire 2 5G 67.8%, Yuneec Typhoon H 2G 38.9%. The two parts of a recording are not consecutive stretches of one session, so this is the nearest the dataset gets to a new recording. It still comes from the same chamber and receiver.

The task is not trivial. I also ran the leaky contrast (`scripts/run_leaky.py`): shuffling all windows 80/20 regardless of time gives **76.3%** (77.5, 75.8, 75.7), no higher than the time-block split. So here, unlike the bearing data in the sibling project, window leakage is **not** inflating the score: 0.55 ms windows are not near-duplicates of their neighbours in a way this CNN exploits, and the error is dominated by something else (section 2). The time-block split was a sound precaution but did not change the answer on this dataset.

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
- **The real weaknesses are two.** (1) Detection coverage for sparse hoppers (Parrot Disco and Mambo, Yuneec Typhoon H): a fixed 0.55 ms window catches a burst only 10-14% of the time, so a deployed system would have to accumulate or use longer windows. (2) Confusion between DJI siblings: Inspire 2, Phantom 4 Pro Plus and Matrice 210 stay at 83-94% even on busy windows. Section 6 below has the confusion matrix and what longer observations do.
- **Gated training helps a little and consistently** (about 3 points on detected windows, same sign in all three seeds), but it is one dataset and one run per seed.
- **The noise floor used by the gate comes from the whole recording.** A live receiver would estimate it from past samples only. Not tested.

## 3. Simulated shift (`scripts/run_shift.py`, figure: `scripts/make_figure.py`)

![Macro accuracy under each simulated impairment](shift.png)

Setup: models trained only on detected windows (1,200 steps, 3 seeds); tested on the 939 detected windows of the held-out time blocks, with one impairment swept at a time. Metric is **macro accuracy** (mean per-drone recall), because the sparse hoppers contribute very few detected windows (13 each for Parrot Disco and Mambo, 35 for Typhoon H), so their share of the average is noisy. Chance is 10%. Every impairment is a *simulation* applied to chamber recordings; none is a field capture. The stated SNR is window mean power over noise power, which counts the recording's own noise floor as signal and averages over the silent gaps of hopping links, so the in-burst SNR is higher than stated.

| Condition | Trained clean | Trained with all impairments | Trained without receiver impairments |
| --- | --- | --- | --- |
| No impairment | **97.4%** | 83.1% | 78.0% |
| SNR 30 dB | 72.9% | 82.8% | 79.2% |
| SNR 20 dB | 47.6% | 79.0% | 79.9% |
| SNR 10 dB | 21.1% | 75.7% | 80.2% |
| SNR 0 dB | 15.2% | 45.6% | 59.2% |
| CFO up to 1 MHz | 84.5% | 79.7% | 76.8% |
| CFO up to 5 MHz | 53.4% | 53.1% | 45.5% |
| I/Q imbalance 2 dB / 15 deg | 71.3% | 79.8% | 78.4% |
| Multipath, decay 0.5, 3-sample (25 ns) spacing | 60.5% | 81.0% | 79.5% |
| Receiver bandwidth 56 MHz | 12.3% | 68.1% | 12.6% |
| Receiver bandwidth 20 MHz | 38.0% | 40.3% | 13.5% |
| ADC 8 bits | 81.0% | 82.2% | 76.5% |
| ADC 6 bits | 38.4% | 72.5% | 61.3% |

(Means over 3 seeds; the full tables are in `results/shift.json`.)

What this says:

- **The clean-trained model is very fragile.** It scores 97% on the unimpaired detected windows and drops to 73% with noise at a nominal 30 dB, 48% at 20 dB and 21% at 10 dB. Multipath alone takes it to about 60%, a narrower receiver to chance, and 6-bit sampling to 38%. Carrier frequency offsets up to 1 MHz cost little (84%), and 12-bit sampling nothing. The collapse at 30 dB suggests the model leans on the chamber's very clean noise floor, but I have not tested that explanation.
- **Training on the impairments recovers most of it, at a price.** With noise from -5 to 30 dB, offsets, imbalance, multipath, narrower receivers and fewer bits in training, accuracy stays around 80% for 10 dB noise, multipath and imbalance. The price is clean accuracy: 83% against 97% (and a wide spread across seeds, 78-89%). Whether a better recipe (more steps, a schedule from easy to hard) closes that gap is untested.
- **Augmentation generalises only partly to impairments it never saw.** The model trained without bandwidth limiting or re-quantisation (third column) is as good as the full-augmentation model on noise, offset, imbalance and multipath, which it did see, but stays at chance when the receiver is narrowed to 56 MHz (12.6%) and recovers only part of the way for 6-bit sampling (61% against 72.5% with it in training and 38% without any augmentation). So training on noise and channel effects did not protect against the receiver changing.
- **A narrower receiver is the most damaging case, and it is not a nuisance to be augmented away.** The hopping drones spread their bursts across the full capture, so a 56 MHz or narrower receiver does not see most of them. Even with that in training, 20 MHz gives only 40%. The clean model's accuracy is not monotonic in bandwidth (12% at 56 MHz, 38% at 20 MHz, 23% at 10 MHz); I do not know why and the windows involved are few.
- **All of this is in-family.** The augmentation and the test impairments come from the same simulator, with overlapping parameter ranges, so the recovery numbers are optimistic about real channels. The unseen-impairment column is the only out-of-family check, and it is weak evidence in the unflattering direction.

## 4. DroneRF: a second dataset (`scripts/run_dronerf.py`)

DroneRF (Al-Sa'd, Allahham, Mohamed, Al-Ali, Khattab, Erbad; Qatar University 2019; CC BY 4.0) is a different team, place and pair of receivers: a Parrot Bebop, a Parrot AR Drone, a DJI Phantom and "no drone" background, recorded in a laboratory with two NI USRP-2943R receivers that each cover half of the 2.4 GHz band (files "L" and "H"). The values are real-valued samples (not IQ). The dataset page does not state the sample rate; 40 MS/s is my inference from 10.25 s of background split into 41 segments of 10,000,000 samples. I stack the two halves as two input channels, use 65,536-sample windows (1.6 ms at that inferred rate), and hold out whole 0.25 s segments. The class sets differ from CardRF's, so no model is transferred between datasets; only the method and its failure modes are compared. Three seeds, 1,200 steps.

**Drone identification** (background / AR / Bebop / Phantom; 140 train, 22 validation, 65 test segments):

| Split | Test accuracy | Macro accuracy |
| --- | --- | --- |
| Ordered: later segments of each class are the test set | 70.3% (70.3, 72.3, 68.3) | 75.2% |
| Interleaved: test segments are neighbours of train segments | 74.9% (75.3, 75.2, 74.2) | 77.2% |

Per class, ordered split: background 100% (all seeds), Phantom 77%, Bebop 66%, AR 58%.

**Drone and flight mode** (10 classes: the five-bit code):

| Split | Test accuracy |
| --- | --- |
| Ordered | **31.2%** (chance 10%) |
| Interleaved | **59.8%** |

Under the ordered split several flight-mode classes are recognised at or below chance (Bebop hovering 2%, Bebop flying 1%, AR on-and-connected 1%, AR hovering 2%, Phantom 71%, background 100%). Interleaving the segments nearly doubles the score, so a large part of the failure is **drift over the recording session**: the signal in later segments differs from earlier ones in a way a model trained on early ones does not carry over. I do not know the cause (a changing lab environment, battery state and flight behaviour are all possibilities). The ordered split is the honest one for "will it work later"; the interleaved split is the optimistic one.

**Noise sweep (drone task, ordered split, macro accuracy):**

| SNR (dB) | Trained clean | Trained with random noise |
| --- | --- | --- |
| none | 75.3% | 74.9% |
| 30 | 75.1% | 75.0% |
| 10 | 74.6% | 75.3% |
| 5 | 74.1% | 76.3% |
| 0 | 61.3% | 74.2% |
| -5 | 33.8% | 45.7% |
| -10 | 25.4% | 26.5% |

This **contrasts sharply with CardRF**, where the clean-trained model collapsed at a nominal 30 dB (73%) and 10 dB (21%). Here the clean-trained model holds until about 5 dB and only noise training buys protection below 5 dB. One reading is that the anechoic CardRF recordings are so clean that a model trained on them never learns to ignore noise, while DroneRF's laboratory recordings already contain real noise, which would make the CardRF fragility a property of unusually clean training data and not of the method. That is consistent with the two sweeps but not proven: the datasets differ in task (10 against 4 classes), signal type (complex IQ against real samples), receiver, SNR definition (window mean power in both, but over different bandwidths) and window length.

## 5. Open-set: recognising a drone the model has never seen (`scripts/run_openset.py`)

The 10 drones are shuffled once into 5 pairs; in each fold one pair is held out as "unknown" (never used in training) and the model is a closed-set classifier over the other 8. Test windows are the detected windows of the held-out time blocks, for known and unknown drones alike. Higher score = more likely known. Five folds x three seeds; the spread below is across those, not a confidence interval.

| Score | AUROC, known vs unknown (mean, min-max over 15 runs) | Unknown windows accepted when 95% of known are accepted (mean) |
| --- | --- | --- |
| Max softmax probability | 0.695 (0.44-0.95) | 82% |
| Energy (logsumexp of logits) | 0.735 (0.40-0.98) | 53% |
| **Mahalanobis distance on the CNN embedding** | **0.935 (0.87-1.00)** | **37%** |

Closed-set accuracy over the 8 known drones in these runs is 95.7% on detected windows; at the 95%-acceptance threshold the accepted known windows are 96.7-98.1% correct depending on the score.

Per fold (mean AUROC; share of unknown accepted at 95% known), Mahalanobis: fold 0 (Mavic Pro + Phantom 4 Pro Plus) 0.89, 71%; fold 1 (Matrice 210 + Parrot Disco) 0.92, 52%; fold 2 (Mavic Mini + Phantom 4) 1.00, 0%; fold 3 (Typhoon H + Inspire 2) 0.87, 59%; fold 4 (Parrot Mambo + Matrice 100) 0.99, 5%.

What this says:

- **Max softmax and energy are not safe for this.** For two held-out DJI models they are *below chance* (AUROC for Phantom 4 Pro Plus 0.27 with MSP, Matrice 210 0.42): the unknown sibling is mapped onto a known DJI model and scored with higher confidence than the known drones themselves. Raw classifier confidence is a poor guide when an unknown looks like a relative of a known class.
- **Distance in feature space works much better**, AUROC 0.87 to 1.00 in every run, and 0.85-0.92 even for the siblings (Phantom 4 Pro Plus 0.85, Matrice 210 0.92) that fooled the other scores.
- **A high AUROC is not a safe operating point.** In the three folds containing a DJI sibling of a known drone, or Inspire 2, the Mahalanobis score still accepts 52-71% of unknown windows when 95% of known windows are kept. Rejecting unknown drones reliably would cost a lot of known windows. Folds with unknowns that look unlike anything known (Mavic Mini, Phantom 4, Mambo) are easy: 0-5% accepted.
- **Unknown drones in the sparse-hopper group are hard to evaluate**: Parrot Disco and Mambo have 13 detected test windows each, so their per-drone numbers (0.96 and 0.98) rest on very little data.
- **Limits.** Single chamber, single receiver; two unknowns per fold; the Mahalanobis fit uses the training windows of the known drones, the gate's noise floor comes from the whole recording, and no shifted conditions were tested here (the shift sweeps show how much a model changes under noise or a narrower receiver, so open-set scores under those conditions are likely worse; not measured).

## 6. Confusion matrix, longer observations, DroneRF bandwidth and bit depth

`scripts/run_burst.py` (CardRF, the gated model, 3 seeds, same time-block split as section 2; `results/burst.json`) and `scripts/run_dronerf_sweeps.py` (`results/dronerf_sweeps.json`).

**Which drones are confused.** Detected test windows only, seeds pooled (each row is a true drone, percentages of its windows):

| True drone | Correct | Largest confusions |
| --- | --- | --- |
| Inspire 2 | 86.5% | Phantom 4 Pro Plus 5.5%, Matrice 100 5.1%, Matrice 210 1.6% |
| Phantom 4 Pro Plus | 86.9% | Matrice 210 5.8%, Inspire 2 3.2%, Matrice 100 2.9% |
| Matrice 210 | 94.5% | Inspire 2 2.7%, Phantom 4 Pro Plus 2.7% |
| Matrice 100 | 97.4% | Mavic Pro 1.5%, Inspire 2 1.0% |
| Mavic Pro | 98.5% | Matrice 100 1.3% |
| Mavic Mini, Phantom 4, Parrot Disco, Parrot Mambo, Yuneec Typhoon H | 100% | none |

So the 83-94% DJI figures of section 2 are mostly a confusion *among Inspire 2, Phantom 4 Pro Plus and Matrice 210* of about 3 to 6% in each direction, with some windows going to Matrice 100 (3 to 5%); Mavic Pro and Matrice 100 are confused only 1 to 1.5% of the time, and Mavic Mini, Phantom 4 and the non-DJI drones are never mistaken for another drone. Why the four are alike I did not investigate (same vendor protocol and similar hardware is the obvious guess, not tested).

**Longer observations.** Consecutive aligned test windows of one recording are grouped into segments of n windows (0.55 ms each); a segment is detected if any window passes the gate, and classified by summing the log-probabilities of its passing windows. Detected share / accuracy among detected / detected and correct, as a share of all segments:

| Drone | 1 window (0.55 ms) | 5 windows (2.8 ms) | 20 windows (11 ms) | 100 windows (55 ms) |
| --- | --- | --- | --- | --- |
| Inspire 2 | 88% / 86% / 76% | 98% / 88% / 87% | 100% / 100% / 100% | 100% / 100% / 100% |
| Phantom 4 Pro Plus | 88% / 87% / 76% | 99% / 89% / 88% | 100% / 100% / 100% | 100% / 100% / 100% |
| Matrice 210 | 87% / 94% / 83% | 99% / 96% / 96% | 100% / 100% / 100% | 100% / 100% / 100% |
| Mavic Mini | 54% / 100% / 54% | 66% / 100% / 66% | 100% / 100% / 100% | 100% / 100% / 100% |
| Parrot Disco | 13% / 100% / 13% | 26% / 100% / 26% | 61% / 100% / 61% | 100% / 100% / 100% |
| Parrot Mambo | 10% / 100% / 10% | 26% / 100% / 26% | 61% / 100% / 61% | 100% / 100% / 100% |
| Yuneec Typhoon H | 14% / 100% / 14% | 35% / 100% / 35% | 75% / 100% / 75% | 89% / 100% / 89% |

(Matrice 100, Mavic Pro and Phantom 4, omitted for space, are at 100% by 20 windows.) Accumulating windows fixes both problems: the DJI confusions disappear within about 11 ms of observation, and the sparse hoppers are caught 61 to 75% of the time within 11 ms and 89 to 100% within 55 ms, always classified correctly when caught. **These longer-segment numbers are thin.** Each segment is a stretch of a single recording in the same test block; a 100-window segment gives only 3 to 9 segments per drone per seed (9 to 27 over three seeds, repeating the same segments), and no 400-window segments exist because the test blocks are too short. Because segments of one recording share the same session and conditions, "100%" here mostly says that errors on single windows are independent enough to vote away within a recording; it does not show that a segment from a *different* recording or day would be classified so cleanly.

**DroneRF bandwidth and bit depth** (real-valued recordings, drone task, ordered split, macro accuracy over 4 classes, chance 25%, mean and standard deviation over 3 seeds). "Augmented" is trained with random noise, band-limiting and quantisation.

| Test condition | Trained clean | Augmented |
| --- | --- | --- |
| clean | 75.7 +- 0.6 | 73.7 +- 2.0 |
| keep the lowest 50% of the spectrum | 32.9 +- 5.3 | 48.9 +- 8.6 |
| lowest 25% | 25.0 +- 0.0 | 29.6 +- 6.5 |
| lowest 12.5% | 25.0 +- 0.0 | 29.4 +- 4.8 |
| lowest 6.25% | 25.0 +- 0.0 | 31.5 +- 6.1 |
| 12 bits | 75.7 +- 0.6 | 73.7 +- 2.0 |
| 8 bits | 75.3 +- 0.9 | 72.4 +- 2.4 |
| 6 bits | 73.5 +- 1.1 | 71.7 +- 2.5 |
| 4 bits | 71.0 +- 0.3 | 70.1 +- 2.8 |
| 3 bits | 68.5 +- 0.5 | 67.6 +- 2.6 |
| 2 bits | 35.7 +- 0.9 | 34.7 +- 0.7 |

- **A narrower receiver is far worse than coarser samples.** Dropping the upper half of the spectrum takes the clean model from 76% to 33%, and below that it predicts a single class (exactly chance). Training with band-limiting recovers part of it at half bandwidth (49%) and almost nothing below. This matches the CardRF finding that augmentation recovers noise and channel effects but not receiver bandwidth.
- **Bit depth hardly matters until 2 bits.** Accuracy falls about 2 points per halving down to 4 bits, 7 points at 3 bits, and collapses at 2 bits. A cheap low-resolution ADC is not the risk; bandwidth is.
- **Only a low-pass was tested** (keep the lowest fraction of the real FFT). Where in the band the drones' energy sits, and whether a band-pass or high-pass cut would be as damaging, I did not test; the collapse at half bandwidth suggests the discriminating content is in the part removed.
- Augmentation here slightly lowers clean accuracy (73.7 against 75.7, within about one standard deviation of the augmented runs).

## Not done yet

An easy-to-hard augmentation schedule and tests of why the clean model is so sensitive to noise; open-set detection under simulated shift or on DroneRF; segment-level decisions across *different* recordings (the data does not allow it: one recording per drone and block); band-pass and high-pass cuts on DroneRF; a cross-dataset test (the class sets do not overlap); why the four DJI models are alike; any real-time or on-device measurement.
