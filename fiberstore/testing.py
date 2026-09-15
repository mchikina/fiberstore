"""Synthetic Fiber-seq BAM with known truth, for tests and downstream fixtures.

Mimics what pbmm2 + fibertools produce: SEQ/CIGAR in reference orientation,
MM/ML and ns/nl/as/al in ORIGINAL-read orientation (reversed for flag-16 reads),
implicit-skip MM, m6A on A+a / T-a channels, CpG 5mC on C+m.
"""
from array import array

import numpy as np

COMP = str.maketrans("ACGT", "TGCA")


def revcomp(s):
    return s.translate(COMP)[::-1]


def _mm_deltas(orig, base, positions):
    """MM-tag delta list for modified `positions` (sorted) among `base` bases of `orig`."""
    idx = np.array([i for i, ch in enumerate(orig) if ch == base], np.int64)
    if len(positions) == 0:
        return []
    ranks = np.searchsorted(idx, np.asarray(positions))
    assert np.array_equal(idx[ranks], positions), "modified position is not a matching base"
    return np.diff(ranks, prepend=-1) - 1


def _read(rng, chrom, ref, s, aligned_target):
    """Simulate one alignment in reference orientation.  Returns dict with q (str),
    q2r (int64 array), cigar (list of (op, n)), nm (int), ins (list of (qstart, len))."""
    q, q2r, cigar, ins = [], [], [], []
    nm = 0
    sl = 0 if rng.random() < 0.7 else int(rng.integers(1, 21))
    if sl:
        q.extend(rng.choice(list("ACGT"), sl))
        q2r.extend([-1] * sl)
        cigar.append((4, sl))
    ref_pos = s
    consumed = 0
    while consumed < aligned_target and ref_pos < len(ref):
        n = int(min(rng.integers(50, 500), len(ref) - ref_pos))
        for j in range(n):
            b = ref[ref_pos + j]
            if rng.random() < 0.01:
                b = rng.choice([c for c in "ACGT" if c != b])
                nm += 1
            q.append(b)
            q2r.append(ref_pos + j)
        cigar.append((0, n))
        ref_pos += n
        consumed += n
        u = rng.random()
        if u < 0.2 and consumed < aligned_target:
            k = int(rng.integers(1, 6))
            ins.append((len(q), k))
            q.extend(rng.choice(list("ACGT"), k))
            q2r.extend([-1] * k)
            cigar.append((1, k))
            nm += k
        elif u < 0.4 and consumed < aligned_target and ref_pos + 6 < len(ref):
            k = int(rng.integers(1, 6))
            cigar.append((2, k))
            ref_pos += k
            nm += k
    sr = 0 if rng.random() < 0.7 else int(rng.integers(1, 21))
    if sr:
        q.extend(rng.choice(list("ACGT"), sr))
        q2r.extend([-1] * sr)
        cigar.append((4, sr))
    return {"q": "".join(q), "q2r": np.array(q2r, np.int64), "cigar": cigar, "nm": nm,
            "ins": ins, "sl": sl, "sr": sr, "ref_end": ref_pos}


def _intervals(rng, L, sl, ins):
    """Disjoint nucleosome and MSP intervals in stored coords; nucleosomes occasionally
    also dropped entirely into a soft clip or insertion (must be lifted away)."""
    nuc, msp = [], []
    if sl >= 5 and rng.random() < 0.5:
        nuc.append((1, 2))                      # inside the left soft clip
    p = sl + int(rng.integers(0, 60))
    while p < L - 60:
        n = int(rng.integers(100, 180))
        if p + n > L:
            break
        nuc.append((p, n))
        p += n
        g = int(rng.integers(10, 120))
        if p + g <= L:
            msp.append((p, g))
        p += g
    for qs, k in ins:
        if k >= 3 and rng.random() < 0.5:
            nuc.append((qs + 1, 1))             # inside an insertion
    nuc.sort()
    return nuc, msp


def _lift_truth(q2r, iv):
    out = []
    for a, l in iv:
        refs = q2r[a:a + l]
        refs = refs[refs >= 0]
        if refs.size:
            out.append((int(refs.min()), int(refs.max()) + 1))
    out.sort()
    return out


def synthetic_bam(path, n_reads=600, seed=0, chroms=None, read_len=(800, 3000)):
    """Write a coordinate-sorted, indexed synthetic BAM to `path`.

    Returns (truth, reference): truth[(qname, reference_start)] holds the fields the
    store must reproduce for every primary alignment; reference = {chrom: sequence}.
    """
    import pysam
    rng = np.random.default_rng(seed)
    chroms = chroms or {"chrT": 30_000, "chrU": 8_000}
    reference = {c: "".join(rng.choice(list("ACGT"), L)) for c, L in chroms.items()}
    header = {"HD": {"VN": "1.6", "SO": "unsorted"},
              "SQ": [{"SN": c, "LN": L} for c, L in chroms.items()],
              "RG": [{"ID": "rg1", "PU": "m1", "SM": "syn"}, {"ID": "rg2", "PU": "m2", "SM": "syn"}]}
    names = list(chroms)
    weights = np.array([chroms[c] for c in names], float)
    truth = {}
    tmp = str(path) + ".unsorted.bam"
    with pysam.AlignmentFile(tmp, "wb", header=header) as out:
        for i in range(n_reads):
            c = names[rng.choice(len(names), p=weights / weights.sum())]
            ref = reference[c]
            target = int(rng.integers(*read_len))
            s = int(rng.integers(0, max(1, len(ref) - target)))
            rd = _read(rng, c, ref, s, target)
            q, q2r, L = rd["q"], rd["q2r"], len(rd["q"])
            reverse = bool(rng.random() < 0.5)
            rg = int(rng.integers(0, 2))
            movie = header["RG"][rg]["PU"]
            qname = f"{movie}/{i}/ccs"

            at = np.array([j for j, ch in enumerate(q) if ch in "AT"])
            m6a_q = np.sort(rng.choice(at, int(0.15 * len(at)), replace=False))
            if reverse:   # C+m calls sit on the original read's C = stored G of a CpG
                cpg = np.array([j for j in range(1, L) if q[j] == "G" and q[j - 1] == "C"], np.int64)
            else:
                cpg = np.array([j for j in range(L - 1) if q[j] == "C" and q[j + 1] == "G"], np.int64)
            c5m_q = np.sort(rng.choice(cpg, int(0.5 * len(cpg)), replace=False)) if len(cpg) else cpg
            c5m_ml = rng.integers(0, 256, len(c5m_q)).astype(np.uint8)
            nuc, msp = _intervals(rng, L, rd["sl"], rd["ins"])

            # tags in original-read orientation
            orig = revcomp(q) if reverse else q
            o = (lambda p: L - 1 - p) if reverse else (lambda p: p)
            m6a_o = np.sort(o(m6a_q))
            a_pos = [p for p in m6a_o if orig[p] == "A"]
            t_pos = [p for p in m6a_o if orig[p] == "T"]
            c_pos = o(c5m_q)
            c_order = np.argsort(c_pos, kind="stable")
            mm = (f"A+a,{','.join(map(str, _mm_deltas(orig, 'A', a_pos)))};"
                  f"T-a,{','.join(map(str, _mm_deltas(orig, 'T', t_pos)))};"
                  f"C+m,{','.join(map(str, _mm_deltas(orig, 'C', c_pos[c_order])))};")
            mm = mm.replace(",;", ";")
            ml = [254] * (len(a_pos) + len(t_pos)) + c5m_ml[c_order].tolist()

            def orient(iv):
                if not reverse:
                    return iv
                return sorted((L - (a + l), l) for a, l in iv)
            nuc_o, msp_o = orient(nuc), orient(msp)

            a = pysam.AlignedSegment(out.header)
            a.query_name = qname
            a.query_sequence = q
            a.flag = 16 if reverse else 0
            a.reference_id = names.index(c)
            a.reference_start = s
            a.mapping_quality = 60
            a.cigartuples = rd["cigar"]
            npass = int(rng.integers(3, 300))
            rq = float(rng.random())
            tags = [("RG", header["RG"][rg]["ID"]), ("np", npass), ("rq", rq, "f"), ("NM", rd["nm"]),
                    ("MM", mm), ("ML", array("B", ml))]
            if nuc_o:
                tags += [("ns", array("i", [x for x, _ in nuc_o])), ("nl", array("i", [x for _, x in nuc_o]))]
            if msp_o:
                tags += [("as", array("i", [x for x, _ in msp_o])), ("al", array("i", [x for _, x in msp_o]))]
            a.set_tags(tags)
            out.write(a)

            c5_ref = [(int(q2r[p]), int(m)) for p, m in zip(c5m_q, c5m_ml) if q2r[p] >= 0]
            c5_ref.sort()
            truth[(qname, s)] = {
                "chrom": c, "start": s, "end": rd["ref_end"], "reverse": reverse, "qlen": L,
                "rg": rg, "movie": ["m1", "m2"].index(movie), "zmw": i,
                "np": min(npass, 255), "rq": rq, "nm": rd["nm"],
                "m6a": np.array(sorted(int(q2r[p]) for p in m6a_q if q2r[p] >= 0), np.int64),
                "c5m": np.array([p for p, _ in c5_ref], np.int64),
                "c5m_ml": np.array([m for _, m in c5_ref], np.uint8),
                "nuc": _lift_truth(q2r, nuc), "msp": _lift_truth(q2r, msp),
            }
            # decoys that must not enter the store
            u = rng.random()
            if u < 0.05:
                b = pysam.AlignedSegment(out.header)
                b.query_name = qname
                b.query_sequence = q[:200]
                b.flag = 2048 | (16 if reverse else 0)
                b.reference_id = names.index(c)
                b.reference_start = max(0, s - 5000)
                b.mapping_quality = 60
                b.cigartuples = [(0, 200)]
                b.set_tags([("RG", header["RG"][rg]["ID"])])
                out.write(b)
            elif u < 0.08:
                b = pysam.AlignedSegment(out.header)
                b.query_name = qname
                b.query_sequence = q[:100]
                b.flag = 256
                b.reference_id = names.index(c)
                b.reference_start = min(len(ref) - 100, s + 5000)
                b.mapping_quality = 0
                b.cigartuples = [(0, 100)]
                b.set_tags([("RG", header["RG"][rg]["ID"])])
                out.write(b)
    pysam.sort("-o", str(path), tmp)
    pysam.index(str(path))
    import os
    os.remove(tmp)
    return truth, reference
