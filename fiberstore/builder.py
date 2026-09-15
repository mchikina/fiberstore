"""BAM -> fiberstore directory.  The only module that imports pysam."""
import glob
import json
import os
import time
import multiprocessing as mp

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .codec import ROW_GROUP, SCHEMA, enc_ints, enc_pos
from .lift import flip_intervals, lift_intervals, query_to_ref

__version_note__ = "uint8 escape-255 deltas"


def fiber_row(r, rg_index, movie_index):
    """One primary pysam AlignedSegment -> dict matching SCHEMA."""
    q2r = query_to_ref(r)
    aligned_q = np.nonzero(q2r >= 0)[0]
    L = r.query_length
    mb = r.modified_bases or {}
    m6a_q, c5m_q, c5m_ml = [], [], []
    for (canon, strand, code), lst in mb.items():
        if code == "a":
            m6a_q.extend(p for p, _ in lst)
        elif code == "m":
            c5m_q.extend(p for p, _ in lst)
            c5m_ml.extend(q for _, q in lst)
    m6a_ref = q2r[np.asarray(m6a_q, np.int64)] if m6a_q else np.zeros(0, np.int64)
    m6a_ref = np.sort(m6a_ref[m6a_ref >= 0])
    if c5m_q:
        c5m_q = np.asarray(c5m_q, np.int64)
        c5m_ml = np.asarray(c5m_ml, np.uint8)
        order = np.argsort(c5m_q, kind="stable")
        c5m_q, c5m_ml = c5m_q[order], c5m_ml[order]
        c5m_ref = q2r[c5m_q]
        ok = c5m_ref >= 0
        c5m_ref, c5m_ml = c5m_ref[ok], c5m_ml[ok]
    else:
        c5m_ref, c5m_ml = np.zeros(0, np.int64), np.zeros(0, np.uint8)

    def ft_intervals(stag, ltag):
        if not r.has_tag(stag):
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        s = np.asarray(r.get_tag(stag), np.int64)
        l = np.asarray(r.get_tag(ltag), np.int64)
        if r.is_reverse:
            s, l = flip_intervals(s, l, L)
        return lift_intervals(q2r, aligned_q, s, l)

    ns, ne = ft_intervals("ns", "nl")
    ms, me = ft_intervals("as", "al")
    movie, zmw = r.query_name.split("/")[:2]
    start = r.reference_start
    return {
        "start": start, "end": r.reference_end, "reverse": r.is_reverse,
        "rg": rg_index[r.get_tag("RG")], "movie": movie_index.setdefault(movie, len(movie_index)),
        "zmw": int(zmw), "qlen": L,
        "np": min(int(r.get_tag("np")), 255) if r.has_tag("np") else 0,
        "rq": float(r.get_tag("rq")) if r.has_tag("rq") else 0.0,
        "nm": min(int(r.get_tag("NM")), 65535) if r.has_tag("NM") else 0,
        "m6a": enc_pos(m6a_ref, start),
        "c5m": enc_pos(c5m_ref, start), "c5m_ml": c5m_ml.astype(np.uint8).tobytes(),
        "nuc_s": enc_pos(ns, start), "nuc_l": enc_ints(ne - ns),
        "msp_s": enc_pos(ms, start), "msp_l": enc_ints(me - ms),
    }


def _write_rows(writer, rows):
    cols = {k: [row[k] for row in rows] for k in SCHEMA.names}
    writer.write_table(pa.table(cols, schema=SCHEMA))


def build_chunk(args):
    """Worker: primary fibers with reference_start in [s, e) of chrom -> one parquet part."""
    import pysam
    bam_path, chrom, s, e, out, rg_index, movies = args
    movie_index = {m: i for i, m in enumerate(movies)}
    bam = pysam.AlignmentFile(bam_path, "rb")
    writer = pq.ParquetWriter(out, SCHEMA, compression="zstd", compression_level=9)
    rows, n = [], 0
    for r in bam.fetch(chrom, s, e):
        if r.is_supplementary or r.is_secondary or r.is_unmapped:
            continue
        if not (s <= r.reference_start < e):
            continue
        rows.append(fiber_row(r, rg_index, movie_index))
        n += 1
        if len(rows) == ROW_GROUP:
            _write_rows(writer, rows)
            rows = []
    if rows:
        _write_rows(writer, rows)
    writer.close()
    if n == 0:
        os.remove(out)
    return chrom, s, n, len(movie_index) > len(movies)


def merge_chrom(args):
    """Stream a chromosome's parts (genomic order) into one file with row groups of
    exactly ROW_GROUP rows (except the last)."""
    outdir, c = args
    parts = sorted(glob.glob(os.path.join(outdir, "parts", f"{c}.*.parquet")))
    if not parts:
        return c, 0
    w = pq.ParquetWriter(os.path.join(outdir, f"{c}.parquet"), SCHEMA,
                         compression="zstd", compression_level=9)
    buf = None
    for p in parts:
        t = pq.read_table(p)
        buf = t if buf is None else pa.concat_tables([buf, t])
        n_full = (buf.num_rows // ROW_GROUP) * ROW_GROUP
        if n_full:
            w.write_table(buf.slice(0, n_full), row_group_size=ROW_GROUP)
            buf = buf.slice(n_full)
    if buf is not None and buf.num_rows:
        w.write_table(buf, row_group_size=ROW_GROUP)
    w.close()
    for p in parts:
        os.remove(p)
    return c, len(parts)


def build(bam_path, outdir, workers=20, chunk=1_000_000, chroms=None, log=print):
    """Build a fiberstore at `outdir` from an indexed, coordinate-sorted BAM.

    Keeps primary alignments only.  `chunk` is the genomic span per worker task.
    Returns the number of fibers written.
    """
    import pysam
    from .store import FiberStore
    log = log or (lambda *a, **k: None)
    bam = pysam.AlignmentFile(bam_path, "rb")
    rgs = [rg["ID"] for rg in bam.header.get("RG", [])]
    rg_index = {k: i for i, k in enumerate(rgs)}
    # movies are the PU field of each RG (qname prefix); order fixes the uint8 index
    movies = []
    for rg in bam.header.get("RG", []):
        if rg.get("PU") and rg["PU"] not in movies:
            movies.append(rg["PU"])
    names = list(chroms) if chroms else list(bam.references)
    os.makedirs(os.path.join(outdir, "parts"), exist_ok=True)
    tasks = []
    for c in names:
        L = bam.get_reference_length(c)
        for s in range(0, L, chunk):
            tasks.append((bam_path, c, s, min(s + chunk, L),
                          os.path.join(outdir, "parts", f"{c}.{s:012d}.parquet"), rg_index, movies))
    bam.close()
    log(f"{len(tasks)} chunks over {len(names)} chromosomes, {workers} workers")
    t0 = time.time()
    done = total = 0
    unseen_movie = False
    ctx = mp.get_context("spawn")   # fork is unsafe once pyarrow's thread pool exists
    with ctx.Pool(workers) as pool:
        for c, s, n, extra in pool.imap_unordered(build_chunk, tasks):
            done += 1
            total += n
            unseen_movie |= extra
            log(f"[{done}/{len(tasks)} {time.time()-t0:7.0f}s] {c}:{s} {n} fibers")
    if unseen_movie:
        log("WARNING: a qname movie was not among the @RG PU fields; movie index unreliable")
    with ctx.Pool(min(workers, 6)) as pool:
        for c, k in pool.imap_unordered(merge_chrom, [(outdir, c) for c in names]):
            log(f"merged {c}: {k} parts")
    os.rmdir(os.path.join(outdir, "parts"))
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump({"source_bam": os.path.abspath(bam_path), "read_groups": rgs, "movies": movies,
                   "row_group": ROW_GROUP, "chroms": names, "encoding": __version_note__,
                   "built": time.strftime("%Y-%m-%d %H:%M")}, f, indent=1)
    FiberStore(outdir)  # writes index.npz
    log(f"done: {total} fibers in {time.time()-t0:.0f}s")
    return total
