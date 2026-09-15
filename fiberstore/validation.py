"""Round-trip check of a store against its source BAM (independent pysam-based lift)."""
import numpy as np

from .store import FiberStore


def _lift(pairs, L, qs, ql):
    """Reference span of each [qs, qs+ql) stored-SEQ interval, via the aligned-pairs dict;
    intervals with no aligned base are dropped.  Deliberately not lift_intervals()."""
    out = []
    for a, l in zip(qs, ql):
        refs = [pairs[q] for q in range(max(a, 0), min(a + l, L)) if q in pairs]
        if refs:
            out.append((min(refs), max(refs) + 1))
    return out


def check_fiber(r, q, i, movies):
    """Compare store row i of query result q with pysam record r.  Returns a list of
    mismatch labels (empty = ok)."""
    bad = []
    pairs = dict(r.get_aligned_pairs(matches_only=True))
    mb = r.modified_bases or {}
    m6a = sorted(pairs[p] for k, v in mb.items() if k[2] == "a" for p, _ in v if p in pairs)
    if not np.array_equal(q["m6a"][i], m6a):
        bad.append("m6a")
    c5 = sorted((pairs[p], ml) for k, v in mb.items() if k[2] == "m" for p, ml in v if p in pairs)
    if not (np.array_equal(q["c5m"][i], [p for p, _ in c5]) and
            np.array_equal(q["c5m_ml"][i], [ml for _, ml in c5])):
        bad.append("c5m")
    for kind, stag, ltag in (("nuc", "ns", "nl"), ("msp", "as", "al")):
        if r.has_tag(stag):
            s, l = np.asarray(r.get_tag(stag)), np.asarray(r.get_tag(ltag))
            if r.is_reverse:
                s = r.query_length - (s + l)
            o = np.argsort(s, kind="stable")
            iv = _lift(pairs, r.query_length, s[o], l[o])
            got = list(zip(q[f"{kind}_s"][i].tolist(), (q[f"{kind}_s"][i] + q[f"{kind}_l"][i]).tolist()))
            if got != iv:
                bad.append(kind)
        elif len(q[f"{kind}_s"][i]):
            bad.append(kind)
    if (q["qlen"][i] != r.query_length or bool(q["reverse"][i]) != r.is_reverse
            or q["end"][i] != r.reference_end):
        bad.append("meta")
    return bad


def validate(bam_path, store_path, n=300, seed=0, region_len=1000, min_frac=0.88,
             min_start=3_000_000, log=print):
    """Random regions: compare selection sets and every decoded field against the BAM.
    Returns True if no problems were found."""
    import pysam
    log = log or (lambda *a, **k: None)
    rng = np.random.default_rng(seed)
    fs = FiberStore(store_path)
    bam = pysam.AlignmentFile(bam_path, "rb")
    chroms = fs.chroms
    lens = np.array([bam.get_reference_length(c) for c in chroms], float)
    n_reg = n_fib = n_bad = 0
    for _ in range(n):
        c = chroms[rng.choice(len(chroms), p=lens / lens.sum())]
        L = bam.get_reference_length(c)
        lo = min(min_start, max(0, L - region_len))
        s = int(rng.integers(lo, max(lo + 1, L - region_len)))
        e = s + region_len
        truth = {}
        for r in bam.fetch(c, s, e):
            if r.is_supplementary or r.is_secondary or r.is_unmapped:
                continue
            if min(r.reference_end, e) - max(r.reference_start, s) >= min_frac * region_len:
                truth[(r.query_name, r.reference_start)] = r
        q = fs.query(c, s, e, min_frac=min_frac)
        keys = [(fs.qname(m, z), int(st)) for m, z, st in zip(q["movie"], q["zmw"], q["start"])]
        n_reg += 1
        if set(keys) != set(truth):
            n_bad += 1
            log(f"selection mismatch {c}:{s}-{e}: store {len(keys)} bam {len(truth)}")
            continue
        for i, k in enumerate(keys):
            n_fib += 1
            bad = check_fiber(truth[k], q, i, fs.meta["movies"])
            if bad:
                n_bad += 1
                log(f"mismatch {k[0]} {c}:{k[1]}: {' '.join(bad)}")
    log(f"{n_reg} regions, {n_fib} fibers checked, {n_bad} problems")
    return n_bad == 0
