# fiberstore

A compact, region-queryable store of single-molecule Fiber-seq data: **one Parquet file**
built once from a fibertools-annotated BAM, read with numpy + pyarrow only. A small sidecar
index (`store.parquet.fsi`, like `.bam.bai`) is created on first open and rebuilt
automatically if it is missing or stale.

On a 187 GB mouse T-cell BAM (12.6 M primary HiFi fibers): 17 GB on disk, "all fibers covering
≥ 88 % of this 1 kb window" in a few ms including decoding, versus 13 ms for an indexed BAM.

```bash
pip install -e .              # reader: numpy, pyarrow
pip install -e ".[build]"     # + pysam, to build from a BAM

fiberstore build    sample.bam sample.parquet --workers 20
fiberstore validate sample.bam sample.parquet --n 300     # random regions vs. the BAM
fiberstore query    sample.parquet chr1:100000000-100001000 --min_frac 0.88
fiberstore info     sample.parquet
fiberstore index    sample.parquet                        # force-rebuild the .fsi sidecar
```

```python
from fiberstore import FiberStore

fs = FiberStore("sample.parquet")
t = fs.query("chr1", 100_000_000, 100_001_000, min_frac=0.88)
t["m6a"][0]                        # int64 reference positions of m6A on the first fiber
t["nuc_s"][0], t["nuc_l"][0]       # nucleosome starts / lengths
t["row"]                           # row indices (within the chromosome), reusable with fetch()

rows = fs.select("chr1", s, e, min_frac=0.88)          # indices only, ~100 µs
t = fs.fetch("chr1", rows, columns=["start", "end", "m6a"])
n = fs.count("chr1", starts, ends, min_frac=0.88)      # one int per region
```

## What is kept

Primary alignments only; one row per fiber, in reference coordinates.

| column | type | content |
| --- | --- | --- |
| `chrom` | string | reference name |
| `start`, `end` | uint32 | reference span `[start, end)` |
| `reverse` | bool | maps to the reverse strand |
| `rg`, `movie` | uint8 | index into the metadata `read_groups` / `movies`; `fs.qname(movie, zmw)` rebuilds the read name |
| `zmw` | uint32 | ZMW hole number |
| `qlen`, `np`, `rq`, `nm` | uint32, uint8, float32, uint16 | read length, CCS passes (capped 255), read quality, edit distance |
| `m6a` | positions | m6A calls, `A+a` and `T-a` channels merged |
| `c5m`, `c5m_ml` | positions, bytes | CpG 5mC calls and their ML byte, in position order |
| `nuc_s`, `nuc_l` | positions, lengths | nucleosomes (fibertools `ns`/`nl`) |
| `msp_s`, `msp_l` | positions, lengths | methylation-sensitive patches (fibertools `as`/`al`) |

Dropped: SEQ, QUAL, CIGAR, per-call m6A ML (fibertools writes a constant), auxiliary tags,
supplementary/secondary alignments, and calls on soft-clipped or inserted bases (they have no
reference coordinate). Intervals are lifted to the span of their first and last aligned base;
an interval with no aligned base is dropped.

Conventions worth knowing:

* Positions are those of the base carrying the call **in the stored (reference-oriented)
  sequence**. For a reverse-strand fiber, a 5mC call therefore sits on the G of the CpG
  (`pos - 1` is the C); m6A calls cover both A and T of the reference on every fiber.
* fibertools' `ns/nl`, `as/al` are in original-read orientation; the build flips them for
  reverse-strand reads, so `nuc_s` is always in reference coordinates.

## File layout

`store.parquet` is an ordinary Parquet file — DuckDB, polars, R `arrow` etc. read it as a
table with the columns above — with three invariants the fast path relies on:

* rows sorted by `(chrom, start)`, chromosomes in the order listed in the metadata;
* row groups of exactly 256 rows that never straddle a chromosome (only the last group of
  each chromosome is partial), zstd-9, statistics kept for `chrom`/`start`/`end` only;
* schema metadata key `fiberstore` holds JSON: `format`, `build_id`, `source_bam`,
  `read_groups`, `movies`, `chroms`, `chrom_rows` (`chrom → [first row group, n row groups,
  n rows]`), `n_fibers`, `built`. `fiberstore.read_meta(path)` returns it.

`store.parquet.fsi` is an `.npz` with the per-chromosome uint32 `start`/`end` arrays (8 bytes
per fiber) and the `build_id` of the file it belongs to. `FiberStore(path)` loads it, or
scans the two columns (~3 s for 12.6 M fibers) and writes it when it is missing, unreadable
or from a different build. Copy the `.parquet` alone and the index regenerates itself.

Queries are two-level: `select` is a `searchsorted` + overlap test on the in-memory index;
`fetch` maps rows to row groups arithmetically (`row // 256`), reads only those groups, and
decodes them. Position and length columns are uint8 escape-255 delta codes
(`fiberstore.codec`), decodable with one `cumsum`.

## Tests

```bash
pip install -e ".[test]" && pytest
```

`fiberstore.testing.synthetic_bam` writes a small indexed BAM with the real tag conventions
(reverse-strand tags in original-read orientation, soft clips, indels, decoy supplementary and
secondary records) together with the exact fields the store must reproduce; the tests build a
store from it and compare every field, check the file-layout invariants and sidecar
regeneration, then run `validate` against the same BAM.

## License

MIT.
