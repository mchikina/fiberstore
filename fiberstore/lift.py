"""Query -> reference coordinate lifting for one aligned read.

Conventions (verified on fibertools 0.3.2 / pbmm2 output):
  * pysam ``modified_bases`` positions are in stored-SEQ orientation.
  * fibertools ``ns/nl`` (nucleosomes) and ``as/al`` (MSPs) are in ORIGINAL-read
    orientation; for reverse-strand reads convert with s' = L - (s + l).
"""
import numpy as np


def query_to_ref(r):
    """q2r[i] = reference position of stored-SEQ base i, or -1 (soft clip / insertion)."""
    q2r = np.full(r.query_length, -1, np.int64)
    q = 0
    ref = r.reference_start
    for op, n in r.cigartuples:
        if op in (0, 7, 8):          # M, =, X
            q2r[q:q + n] = np.arange(ref, ref + n)
            q += n
            ref += n
        elif op in (1, 4):           # I, S
            q += n
        elif op in (2, 3):           # D, N
            ref += n
    return q2r


def flip_intervals(s, l, L):
    """Original-read -> stored-SEQ orientation for a reverse-strand read, re-sorted."""
    s = L - (np.asarray(s, np.int64) + np.asarray(l, np.int64))
    o = np.argsort(s, kind="stable")
    return s[o], np.asarray(l, np.int64)[o]


def lift_intervals(q2r, aligned_q, qs, ql):
    """Lift [qs, qs+ql) intervals in stored-SEQ coords to reference [s, e).

    An interval maps to the span from its first to its last aligned base; intervals
    containing no aligned base (entirely inside an insertion or a soft clip) are dropped.
    `aligned_q` = sorted stored-SEQ indices with q2r >= 0.
    """
    qs = np.asarray(qs, np.int64)
    ql = np.asarray(ql, np.int64)
    if qs.size == 0 or aligned_q.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    j0 = np.searchsorted(aligned_q, qs, "left")            # first aligned base >= qs
    j1 = np.searchsorted(aligned_q, qs + ql, "left") - 1   # last aligned base < qs + ql
    ok = j1 >= j0
    j0, j1 = j0[ok], j1[ok]
    return q2r[aligned_q[j0]], q2r[aligned_q[j1]] + 1
