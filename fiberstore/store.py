"""Read side: in-memory interval index + row-group-granular Parquet fetch.  No pysam."""
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .codec import BIN_COLS, LEN_COLS, MAX_READ_LEN, POS_COLS, SCHEMA, dec_len, dec_pos


class FiberStore:
    """A fiberstore directory.  Opening loads (or builds) `index.npz`: per-chromosome
    uint32 start/end arrays, ~8 bytes per fiber in memory."""

    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        self.files = {c: os.path.join(path, f"{c}.parquet") for c in self.meta["chroms"]
                      if os.path.exists(os.path.join(path, f"{c}.parquet"))}
        self._pf = {}
        idx = os.path.join(path, "index.npz")
        if os.path.exists(idx):
            z = np.load(idx)
            self.start = {c: z[f"{c}_s"] for c in self.files}
            self.end = {c: z[f"{c}_e"] for c in self.files}
        else:
            self.start, self.end = {}, {}
            for c, f in self.files.items():
                t = pq.read_table(f, columns=["start", "end"])
                self.start[c] = t["start"].to_numpy().astype(np.uint32)
                self.end[c] = t["end"].to_numpy().astype(np.uint32)
            np.savez(idx, **{f"{c}_s": v for c, v in self.start.items()},
                     **{f"{c}_e": v for c, v in self.end.items()})

    def __len__(self):
        return sum(len(v) for v in self.start.values())

    @property
    def chroms(self):
        return list(self.files)

    @property
    def columns(self):
        return list(SCHEMA.names)

    def qname(self, movie, zmw):
        """Reconstruct the BAM query name from the movie index and ZMW."""
        return f"{self.meta['movies'][int(movie)]}/{int(zmw)}/ccs"

    # ------------------------------------------------------------------ selection
    def select(self, chrom, s, e, min_frac=0.0, min_overlap=1):
        """Row indices (into chrom's table, ascending) of fibers overlapping [s, e) by
        at least max(min_overlap, min_frac * (e - s)) bases.  Pure numpy on the index."""
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
    def _file(self, chrom):
        """ParquetFile plus the cumulative row offset of each row group."""
        if chrom not in self._pf:
            pf = pq.ParquetFile(self.files[chrom])
            md = pf.metadata
            sizes = np.array([md.row_group(i).num_rows for i in range(md.num_row_groups)], np.int64)
            self._pf[chrom] = (pf, np.concatenate([[0], np.cumsum(sizes)]))
        return self._pf[chrom]

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
        pf, offsets = self._file(chrom)
        grp = np.searchsorted(offsets, rows, "right") - 1        # row group of each row
        groups = np.unique(grp)
        tab = pf.read_row_groups(groups.tolist(), columns=read_cols)
        local = np.concatenate([[0], np.cumsum(offsets[groups + 1] - offsets[groups])])
        base = local[np.searchsorted(groups, grp)] + (rows - offsets[grp])
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
