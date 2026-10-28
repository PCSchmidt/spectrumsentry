# SpectrumSentry: Evidence Notes

Companion to spectrumsentry.md. Labels: Confirmed, Ran it, Per the docs, My reading, Ask the author, Author says.

## E1 Fact sheet

| Fact | Value | Label |
| --- | --- | --- |
| Name, licence | SpectrumSentry v0.1.0, MIT; data CC BY 4.0 (CardRF, DroneRF) | Confirmed (LICENSE, README, pyproject) |
| Kind | Measurement study plus small library: spectrogram CNN, simulated shift, open-set scores | Confirmed |
| Status | "early" per README; three README goals all measured | Per the docs, Confirmed (results files) |
| Size | about 2,140 lines of Python: 12 library files (~730 lines), 8 experiment scripts plus 3 helper scripts, 6 test files | Confirmed (wc -l) |
| Tests | 37 test functions in 6 files; 37 passed | Ran it |
| Commits | 8, 2026-10-17 to 2026-10-23, all authored as Chris Schmidt; 6 on 2026-10-17 | Confirmed (git log) |
| Visibility | public repository | Ran it (gh repo view) |
| Data | CardRF about 8.6 GB; DroneRF 3.75 GB zip; neither committed (.gitignore: data, bin, npy, zip) | Per the docs, Confirmed (.gitignore) |
| Paid services | none; no network calls in the library; download script fetches CardRF | Confirmed (grep of src) |
| Deployed / users | none evidenced | Confirmed (no deploy config) / Ask the author |
| Tool credit | The history carries a co-author credit on 8 commits; per the standing instruction the tool is not named. Treated as "AI coding agents". | Confirmed |

## E2 Reading and running log

Read: README, docs/design.md, docs/results.md, SECURITY.md, pyproject.toml, ci.yml, all 12 library files, all 6 tests (gate, openset, augment in full; others skimmed), script argument defaults, results JSON structure.
Not read in full: the experiment scripts' bodies, dronerf.py beyond its header and split docstring, the figure script.
Not run: experiment scripts (need data and GPU), download scripts.

Commands run (all in a scratch clone outside the original clone; the original clone was only read):
- `pytest -q -p no:cacheprovider tests` with PYTHONPATH=src, using an existing local Python environment that already had torch and pytest: 37 passed in 2.85 s. Ran it.
- A probe script on toy inputs (E4).
- A parameter count of the network from the code: 242,250 parameters for 10 classes (241,764 for the DroneRF 2-channel, 4-class variant). Ran it.
- `gh run list`: latest CI run on main (2026-10-23, 43 s) completed successfully; the one before it (2026-10-17) also success; a Dependency Graph run also success. No red CI. Ran it.
- Results JSON recomputed: closed-set test per seed 77.28, 77.02, 77.28 (mean 77.2); cross-recording 67.1, 67.1, 70.0 (68.1); open-set AUROC means over 15 runs: max-softmax 0.695 (0.44-0.95), energy 0.735 (0.40-0.98), Mahalanobis 0.935 (0.87-1.00); unknown accepted at 95% known: 82%, 53%, 37%; gated.json 6,527 test windows, 4,026 detected (61.7%); leaky split 77.46, 75.79, 75.70 (mean 76.3); shift.json clean-trained macro accuracy 97.3% undamaged, 21.1% at 10 dB. Ran it (L3 for these).
No secrets were read; no .env exists in the repo listing. Final state of the original clone: no tracked-file changes.

## E3 Claims check (selected)

| ID | Claim | Level | Verdict |
| --- | --- | --- | --- |
| CL-01 | 77% on held-out time blocks, 68% on held-out recording | L3 | Supported |
| CL-02 | 93% on windows that contain a burst (README) | L1/L3 | Partly: 92.8% is for the network trained only on gated windows; the network trained on all windows gets 89.6% |
| CL-03 | clean-trained model falls from 97% to 21% at 10 dB | L3 | Supported (macro accuracy on detected windows, 1,200 steps) |
| CL-04 | AUROC 0.94 vs softmax 0.70 | L3 | Supported (0.935, 0.695) |
| CL-05 | accepts 37% of unknown windows at a 95%-known threshold | L3 | Supported (mean over 15 runs) |
| CL-06 | random window split 76% vs time-block 77% | L3 | Supported (leaky.json) |
| CL-07 | gate thresholds "fixed before looking at results" | none | Not checkable from history (E8) |
| CL-08 | Window length 0.55 ms | Code | Partly: true at 120 MS/s only; at 200 MS/s (5.8 GHz files) 65,536 samples is 0.33 ms |
| CL-09 | DroneRF sample rate 40 MS/s | Per the docs | Inference by the author, stated as such |
| CL-10 | Longer-segment table (11 ms, 55 ms) | L1 only | Matches results text; script not re-run; thin per the docs themselves |

## E4 Rule attacks (probes in a scratch copy, toy inputs)

- parse(Path("foo.bin")): ValueError with a clear message.
- Gate({'other.bin':1}).mask on a recording not in its table: KeyError.
- spectrogram of 100 samples (shorter than one 256-sample slice): RuntimeError from the FFT library.
- spectrogram of an all-zero window: finite output.
- Mahalanobis fitted with an empty class: means for that class are NaN, no error, so its distances are NaN and the nearest-class minimum is not guaranteed meaningful.
- awgn and quantize on zero windows: no crash, output zeros.
- auroc with an empty known set: NaN with a runtime warning.
Verdict: all concern inputs the scripts never produce; Low. Ran it.

## E5 Code map

Library (src/spectrumsentry): catalog (parse CardRF file names), iq (read int16 IQ, cut windows), data (splits: train 0-60%, val 65-75%, test 80-100%; sampling), gate (energy gate, noise floor), spec (spectrogram), model (SpecCnn: four Conv-BatchNorm-ReLU-MaxPool blocks of width 32, 64, 128, 128, then pooling, dropout 0.2, linear), train (AdamW, learning rate 2e-3, OneCycle schedule, batch 48, balanced class sampling), impair and augment (simulators and random augmenter), openset (msp, energy, Mahalanobis with ridge, auroc, threshold at 95% known), dronerf (second dataset). Scripts: one per experiment plus download and convert helpers. Defaults: 600 steps for closed-set, gated, leaky, burst; 1,200 for shift, open-set, DroneRF runs. Confirmed.

## E6 Model call sites

One learned model, trained from scratch, no pretrained weights, no external model calls, no cost. The embedding used for open-set is the 128-number pooled output before dropout and the final linear layer. Confirmed (model.py).

## E7 Tests and CI

37 pass. CI: ubuntu, Python 3.10 and 3.12, CPU torch, `pytest -q` only. No lint, type check or experiment run. Coverage not measured. Spot checks: test_gate burst-vs-noise asserts the mask and an occupancy between 0.4 and 0.6; test_openset pairwise-definition test recomputes AUROC independently by pair counting; test_augment bandlimit test checks a 20 MHz tone survives and a 40 MHz tone does not under per-window sample rates. These assert behaviour. Ran it / Confirmed.

## E8 Plan versus results, history

Eight commits over six days; six on 2026-10-17 including scaffold, baseline, leaky contrast, shift sweeps, DroneRF and open-set. The README's note that the gate thresholds were fixed before looking at results cannot be verified from this history. One commit message says it corrects docs that assumed leakage mattered. Tool credit appears on 8 commits of 8 (not named here by instruction). Confirmed.

## E9 Gates and approvals

None.

## E10 Environment

pyproject: numpy>=1.26, torch>=2.2, pytest>=8, matplotlib>=3.8 optional; no lockfile; results files do not record library versions. Saved closed-set results record device "cuda". Confirmed.

## E11 Security and privacy

Public data only; SECURITY.md describes private vulnerability reporting. No secret scan of history was run beyond reading file lists (My reading: nothing needed; no keys or services). No statement about live security state is made.

## E12 Shortcomings register (summary)

| ID | Shortcoming | Severity | Disclosed | Options |
| --- | --- | --- | --- | --- |
| S1 | One chamber, one receiver, about 1 s per drone | High | Yes | Add a second receiver's data; or keep scope statement |
| S2 | Gate noise floor uses whole recording | Medium | Yes | Causal estimate from past samples; or document |
| S3 | Impairments simulated, in-family augment and test | Medium | Yes | Held-out impairment families; real captures |
| S4 | No version pins, results do not record versions | Low | No | Lockfile; version stamp in JSON |
| S5 | CI never checks results | Medium | No | Small fixture-based result test |
| S6 | Stale wording in README and design notes | Low | No | Edit |
| S7 | No calibration check | Low | n/a | Add reliability measure |
| S8 | Edge-case crashes in E4 | Low | No | Validate inputs |

## E13 Math and method

Mahalanobis: d(x) = min over classes c of (x - mu_c)^T S^-1 (x - mu_c), S the shared covariance of training embeddings plus a ridge of 1e-3 times its average diagonal; score = -d. AUROC by rank sums with average ranks for ties. Threshold at 95% known by the 5th percentile of known scores. Noise: SNR in dB = 10 log10(mean window power / noise power), so the recording's own noise counts as signal. Spectrogram: Hann-windowed 256-point FFT, non-overlapping, log10(power + 1e-12), adaptive average pool to 64 x 64, per-example standardisation. References in code: Hendrycks and Gimpel 2017, Liu et al. 2020, Lee et al. 2018.

## E14 Discrepancies

- results.md and the README give 0.55 ms for the window; 0.33 ms for 5.8 GHz files.
- design.md heading "Simulated shift (planned)"; README "What it will do" lists items already done.
- README "93% on windows that contain a burst" is the gated-trained network.
- Step counts differ between experiments (E5), so cross-experiment numbers use different training lengths.

## E15 Production readiness

| Trait | Rating | Evidence |
| --- | --- | --- |
| Clean-start run | Partly | install and run commands in README; data download manual for DroneRF; no pins |
| Checks block a bad change | Partly | CI runs unit tests only |
| Quality measured against baseline | Yes | chance baselines, leaky contrast, seeds |
| Safe on failure | No | no timeouts; not a service |
| Human in the loop | Not applicable | |
| Prompts, models versioned | Not applicable / No | no saved model files |
| Logs and monitoring | No | |
| Data and secrets | Partly | public data only, no secrets needed |
Placement: works locally.

## E16 Review dispositions

Pending (no review pass yet).

## E17 Completeness

| Promised | Status |
| --- | --- |
| Closed-set classifier with time-block and cross-recording splits | Done and tested (unit tests for splits; results recorded) |
| Simulated shift sweeps and augmentation | Done and measured |
| Open-set detection | Done and measured |
| Second dataset | Done |
| Real-time or on-device measurement | Planned only (results write-up "not done yet") |
| Cross-dataset test | Not possible with these class sets |
Backlog order: tidy docs and pin versions (small, medium payoff); result check in CI (small, medium); different-receiver test (large, high).

## Diagram sources

Three images, drawn with matplotlib in a scratch environment: spectrumsentry-learned-vs-rules.png, spectrumsentry-parts.png, spectrumsentry-results.png (values copied from the shift table in the results write-up; checked against shift.json means).

## Revision log (v2 revision after reader check)

Previous text archived as spectrumsentry.v1draft.md (4,292 words); revised guide is about 3,560 words (plain wc -w). All ten reader-check fixes applied.

1. Window and burst defined in the second opening paragraph (and in the glossary).
2. Opening arithmetic (10 dB as a tenth of power) cut; headline numbers kept to 77%, about 93% on burst windows, 97% to 21% at 10 dB.
3. Deterministic-versus-probabilistic paragraph moved to third paragraph (within the first ~150 words of body); "calibrated probabilities" wording replaced.
4. Methods chapter simplified: softmax, energy score, Mahalanobis (toy example kept), AUROC (with a felt scale: 0.9 and above is very good) and I/Q imbalance each explained in one plain sentence; three-score detail shortened.
5. SNR, closed-set, bandwidth, macro accuracy, seed, augmentation defined at first use; "damage" now used in text and charts (chart said "impairments").
6. AUROC given a felt scale (0.5 coin flip, 0.9 and above very good; 0.935 versus 0.695 stated in the results).
7. Edge-cases chapter cut to a short paragraph of tests, odd inputs (no crash catalogue, no NaN detail) and splits.
8. "Where the errors are" rewritten as a short story (silence first, then DJI confusions, then listening longer).
9. Removed "Growth is not the issue", "A gap / A deliberate limit" labels, the bearing-vibration aside, the sample-rate inference and the unexplained "three families" (target roles named instead).
10. Recruiter section: ten questions folded into five, skeptic list into one question; glossary extended (window, burst, SNR, bandwidth, closed-set, macro accuracy, seed).

Doc-versus-code caveats each stated once, in the results chapter: 0.55 ms window only for 2.4 GHz files; README 93% is the network trained on gated windows only; README and design notes describe finished work as planned. Building tools named only as "AI coding agents". Results-chart values re-read from the previous chart (which was copied from the shift table; clean 97, 10 dB 21 etc. unchanged); text figures rounded from E3 values. All three images regenerated (matplotlib in a scratch venv), labels changed (impairments to damage, macro accuracy and SNR explained, "feature vector" to "internal summary", "half-millisecond" dropped from the title); viewed, no clipped or overlapping text.
