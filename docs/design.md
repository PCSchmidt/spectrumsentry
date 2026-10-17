# Design and scope

## Receive-only

This project only analyses recordings of radio signals. It never transmits, and nothing in it is a transmitter, jammer or spoofer. Transmitting, jamming or spoofing without a licence is illegal in most places. If this is ever extended to live capture, that capture must be passive reception on a receive-only radio.

## What the data is, and what it cannot show

- **CardRF** (Tampere University; Zenodo record 4264467; CC BY 4.0): raw IQ (16-bit interleaved) of the control and video links of 10 consumer drones. Control hopping is captured at 2.44 GHz (120 MS/s, 1 second) and 5.8 GHz (200 MS/s, 0.5 seconds) where a drone supports the band. Recorded in an anechoic chamber with the drone on a turntable and the controller beside it.
- It is one second per drone per band, one chamber, one receiver. So it **cannot** tell us how a classifier behaves outdoors, with another receiver, at range, or with other emitters present. Anything about those is a *simulation* applied to these clean recordings (`spectrumsentry.impair`), labelled as such wherever it appears.

## Splits

Windows from one recording could be near-duplicates (a hopping pattern repeats), so a random window split might measure memorisation, as it did on the bearing data in the sibling project. I checked: on this dataset a random window split scores 76%, the same as the time-block split (77%), so leakage does not matter here (see results.md). The time-block split is kept as a precaution. Each recording is cut into contiguous time blocks with gaps (train 0-60%, validation 65-75%, test 80-100%). Drones recorded in two files give a cross-recording test: train on part 1, test on part 2. Only four drones have this.

## Examples and model

A window is 65,536 IQ samples. Its spectrogram is non-overlapping 256-point FFTs, log power, DC centred, pooled to 64 x 64 and standardised per example, then a four-block CNN. Per-example standardisation removes absolute level, so receiver gain differences do not show up.

## Simulated shift (planned)

Added noise (stated SNR counts the recording's own noise floor as signal, so true SNR is lower), carrier frequency offset, I/Q imbalance, multipath, a narrower receiver (brick-wall band limit), and fewer ADC bits. These are crude stand-ins. The aim is to measure how fast a model trained on clean recordings degrades, and how much training on the same impairments recovers, not to predict field accuracy.

## Open decisions

- Whether DroneRF (a different team, place and receiver) can serve as an independent test set; the class sets do not overlap with CardRF, so it would test the *method* and the shift, not the same classifier.
- How to define "unknown drone" for the open-set test (hold out whole drone models).
