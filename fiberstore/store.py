"""Read side: one Parquet file + sidecar interval index.  No pysam.

Layout invariants (written by builder.build, relied on here):
  * rows sorted by (chrom, start) in the chromosome order of the metadata;
  * row groups of exactly ROW_GROUP rows, never straddling a chromosome, so the
    last group of each chromosome is the only partial one;
  * the JSON under schema-metadata key META_KEY records, per chromosome, its first
    row group, number of row groups and number of rows.
"""
import json
import os
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .codec import (BIN_COLS, INDEX_SUFFIX, LEN_COLS, MAX_READ_LEN, META_KEY, POS_COLS,
                    ROW_GROUP, SCHEMA, dec_len, dec_pos)


def read_meta(path, schema=None):
    """The fiberstore metadata dict of a store file (or of an already-read schema)."""
    md = (schema or pq.read_schema(path)).metadata or {}
    if META_KEY not in md:
        raise ValueError(f"{path}: not a fiberstore file (no {META_KEY.decode()} metadata)")
    return json.loads(md[META_KEY])


class FiberStore:
    """A fiberstore file.  Opening loads the sidecar index `<path>.fsi` (per-chromosome
    uint32 start/end arrays, 8 bytes per fiber in memory) or rebuilds it from the
    start/end columns when it is missing or belongs to a different build."""

    def __init__(self, path, index=True):
        self.path = path
        self._pf = pq.ParquetFile(path)
        self.meta = read_meta(path, self._pf.schema_arrow)   # footer parsed once
        # chrom -> (first row group, n row groups, n rows)
        self._rg = {c: tuple(v) for c, v in self.meta["chrom_rows"].items()}
        for c, (rg0, nrg, n) in self._rg.items():
            if nrg != -(-n // ROW_GROUP):
                raise ValueError(f"{path}: {c} has {nrg} row groups for {n} rows")
        self.start, self.end = {}, {}
        if index:
            self._load_index()

    # ------------------------------------------------------------------ index
    @property
    def index_path(self):
        return self.path + INDEX_SUFFIX

    def _load_index(self):
        p = self.index_path
        if os.path.exists(p):
            try:
                with np.load(p) as z:
                    if str(z["build_id"]) == self.meta["build_id"]:
                        self.start = {c: z[f"{c}_s"] for c in self._rg}
                        self.end = {c: z[f"{c}_e"] for c in self._rg}
                        return
            except (KeyError, OSError, ValueError):
                pass
        self.build_index()

    def build_index(self, write=True):
        """Rebuild the in-memory index from the start/end columns (a scan of two uint32
        columns over the whole file) and, if `write`, save it as the sidecar."""
        t = self._pf.read(columns=["start", "end"])
        st = t["start"].to_numpy().astype(np.uint32)
        en = t["end"].to_numpy().astype(np.uint32)
        row = 0
        for c, (rg0, nrg, n) in self._rg.items():
            self.start[c], self.end[c] = st[row:row + n], en[row:row + n]
            row += n
        if write:
            try:
                with open(self.index_path, "wb") as f:   # file object: no ".npz" appended
                    np.savez(f, build_id=np.array(self.meta["build_id"]),
                             **{f"{c}_s": v for c, v in self.start.items()},
                             **{f"{c}_e": v for c, v in self.end.items()})
            except OSError as e:
                warnings.warn(f"could not write {self.index_path}: {e}")

    # ------------------------------------------------------------------ basics
    def __len__(self):
        return sum(v[2] for v in self._rg.values())

    @property
    def chroms(self):
        return list(self._rg)

    @property
    def columns(self):
        return list(SCHEMA.names)

    def qname(self, movie, zmw):
        """Reconstruct the BAM query name from the movie index and ZMW."""
        return f"{self.meta['movies'][int(movie)]}/{int(zmw)}/ccs"

    # ------------------------------------------------------------------ selection
    def select(self, chrom, s, e, min_frac=0.0, min_overlap=1):
        """Row indices (within chrom, ascending) of fibers overlapping [s, e) by at
        least max(min_overlap, min_frac * (e - s)) bases.  Pure numpy on the index."""
        st, en = self.start[chrom], self.end[chrom]
        i0 = np.searchsorted(st, max(0, s - MAX_READ_LEN), "left")
        i1 = np.searchsorted(st, e, "left")
        ov = np.minimum(en[i0:i1].astype(np.int64), e) - np.maximum(st[i0:i1].astype(np.int64), s)
        need = max(min_overlap, min_frac * (e - s))
        return i0 + np.nonzero(ov >= need)[0]

    def count(self, chrom, starts, ends, min_frac=0.0, min_overlap=1):
        """Number of qualifying fibers for each region."""
        return np.array([len(self.select(chrom, s, e, min_frac, min_overlap))
                         for s, e in zip(starts, ends)], np.int64)

    # ------------------------------------------------------------------ fetching
    def fetch(self, chrom, rows, columns=None, decode=True):
        """Fetch rows (indices from `select`) as {column: array-or-list}.

        Scalar columns come back as numpy arrays; encoded columns as lists of int64
        arrays in reference coordinates (positions) or lengths, or raw bytes if
        decode=False.  Only the row groups holding the requested rows are read.
        """
        rows = np.asarray(rows, np.int64)
        cols = list(columns) if columns else list(SCHEMA.names)
        need_start = decode and any(c in POS_COLS for c in cols)
        read_cols = list(dict.fromkeys((["start"] if need_start else []) + cols))
        if rows.size == 0:
            return {c: ([] if c in BIN_COLS else
                        pa.array([], SCHEMA.field(c).type).to_numpy(zero_copy_only=False))
                    for c in cols}
        rg0, nrg, n = self._rg[chrom]
        if rows.min() < 0 or rows.max() >= n:
            raise IndexError(f"row out of range for {chrom} ({n} rows)")
        grp = rows // ROW_GROUP                      # row group of each row, within chrom
        groups = np.unique(grp)
        tab = self._pf.read_row_groups((rg0 + groups).tolist(), columns=read_cols)
        # groups come back in ascending order and only the chromosome's last group can
        # be partial, so every group but the last of `groups` spans exactly ROW_GROUP rows
        base = np.searchsorted(groups, grp) * ROW_GROUP + rows % ROW_GROUP
        tab = tab.take(pa.array(base))
        out = {}
        starts = tab["start"].to_numpy() if need_start else None
        for c in cols:
            col = tab[c]
            if c in BIN_COLS and decode:
                bufs = col.to_pylist()
                if c in POS_COLS:
                    out[c] = [dec_pos(b, int(o)) for b, o in zip(bufs, starts)]
                elif c in LEN_COLS:
                    out[c] = [dec_len(b) for b in bufs]
                else:
                    out[c] = [np.frombuffer(b, np.uint8) for b in bufs]
            elif c in BIN_COLS:
                out[c] = col.to_pylist()
            else:
                out[c] = col.to_numpy(zero_copy_only=False)
        return out

    def query(self, chrom, s, e, min_frac=0.0, columns=None, min_overlap=1):
        """select + fetch; the result also carries the row indices under 'row'."""
        rows = self.select(chrom, s, e, min_frac, min_overlap)
        out = self.fetch(chrom, rows, columns)
        out["row"] = rows
        return out

    def intervals(self, chrom, rows, kind="nuc"):
        """(starts, ends) list pairs for nucleosomes ('nuc') or MSPs ('msp')."""
        t = self.fetch(chrom, rows, [f"{kind}_s", f"{kind}_l"])
        return t[f"{kind}_s"], [s + l for s, l in zip(t[f"{kind}_s"], t[f"{kind}_l"])]
