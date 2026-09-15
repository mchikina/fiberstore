from types import SimpleNamespace

import numpy as np

from fiberstore.lift import flip_intervals, lift_intervals, query_to_ref


def _seg(cigar, start=100):
    L = sum(n for op, n in cigar if op in (0, 1, 4, 7, 8))
    return SimpleNamespace(cigartuples=cigar, reference_start=start, query_length=L)


def test_query_to_ref_ops():
    #  3S 4M 2I 3M 2D 2M 1S  -> query 15 bases
    r = _seg([(4, 3), (0, 4), (1, 2), (0, 3), (2, 2), (0, 2), (4, 1)])
    q2r = query_to_ref(r)
    assert q2r.tolist() == [-1, -1, -1, 100, 101, 102, 103, -1, -1, 104, 105, 106, 109, 110, -1]


def test_lift_basic_and_dropped():
    r = _seg([(4, 3), (0, 4), (1, 2), (0, 3), (2, 2), (0, 2), (4, 1)])
    q2r = query_to_ref(r)
    aq = np.nonzero(q2r >= 0)[0]
    qs = np.array([0, 3, 5, 7, 9, 11, 14])
    ql = np.array([2, 2, 4, 2, 4, 3, 1])
    s, e = lift_intervals(q2r, aq, qs, ql)
    # [0,2) soft clip only -> dropped; [3,5) -> 100..102; [5,9) -> 102..104 (spans insertion);
    # [7,9) insertion only -> dropped; [9,13) -> 104..110 (spans deletion);
    # [11,14) -> 106..111; [14,15) right clip -> dropped
    assert s.tolist() == [100, 102, 104, 106]
    assert e.tolist() == [102, 104, 110, 111]


def test_lift_empty():
    q2r = np.array([-1, -1])
    s, e = lift_intervals(q2r, np.nonzero(q2r >= 0)[0], [0], [2])
    assert s.size == 0 and e.size == 0
    s, e = lift_intervals(np.array([5]), np.array([0]), [], [])
    assert s.size == 0


def test_flip_intervals():
    s, l = flip_intervals([0, 50], [10, 20], 100)
    assert s.tolist() == [30, 90] and l.tolist() == [20, 10]
