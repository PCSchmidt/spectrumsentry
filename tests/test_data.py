from pathlib import Path

import numpy as np

from spectrumsentry.catalog import Recording
from spectrumsentry.data import BLOCKS, WINDOW, block_spans, class_names, eval_windows, sample_train, whole_spans


def rec(drone, part=0, band="2G"):
    return Recording(Path(f"{drone}_{band}.bin"), drone, band, "", part)


RECS = [rec("A"), rec("B", 1), rec("B", 2), rec("C")]
NAMES = class_names(RECS)


def test_blocks_are_ordered_and_separated_by_gaps():
    tr, va, te = (BLOCKS[k] for k in ("train", "val", "test"))
    assert tr[1] < va[0] and va[1] < te[0]


def test_class_names_merge_parts_of_one_drone():
    assert NAMES == ["A", "B", "C"]


def test_spans_never_overlap_between_blocks():
    for r in range(len(RECS)):
        spans = {b: block_spans([RECS[r]], NAMES, b)[0] for b in BLOCKS}
        assert spans["train"].hi < spans["val"].lo and spans["val"].hi < spans["test"].lo


def test_parts_filter_gives_a_cross_recording_split():
    train = block_spans(RECS, NAMES, "train", parts=(0, 1))
    assert {s.rec.part for s in train} == {0, 1}
    held_out = whole_spans(RECS, NAMES, parts=(2,))
    assert len(held_out) == 1 and held_out[0].rec.drone == "B" and held_out[0].lo == 0


def test_training_samples_stay_inside_their_span_and_balance_classes():
    spans = block_spans(RECS, NAMES, "train", parts=(0, 1))
    items = sample_train(spans, 3000, np.random.default_rng(0))
    assert all(s.lo <= start and start + WINDOW <= s.hi for s, start in items)
    counts = np.bincount([s.label for s, _ in items], minlength=3)
    assert counts.min() > 800  # about 1000 each, whatever the number of files per class


def test_eval_windows_are_aligned_distinct_and_capped():
    span = block_spans([RECS[0]], NAMES, "test")[0]
    items = eval_windows([span], 10)
    starts = [st for _, st in items]
    assert len(starts) == 10 and len(set(starts)) == 10
    assert all((st - span.lo) % WINDOW == 0 and st + WINDOW <= span.hi for st in starts)
