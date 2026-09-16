"""fiberstore command line.

    fiberstore build    BAM STORE.parquet [--workers 20] [--chunk 1000000] [--chroms chr1,chr2]
    fiberstore index    STORE.parquet            rebuild the .fsi sidecar
    fiberstore validate BAM STORE.parquet [--n 300]
    fiberstore query    STORE.parquet chr1:100000000-100001000 [--min_frac 0.88]
    fiberstore info     STORE.parquet
"""
import argparse
import sys
import time


def parse_region(region):
    c, rng = region.rsplit(":", 1)
    s, e = (int(x.replace(",", "")) for x in rng.split("-"))
    return c, s, e


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fiberstore", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="BAM -> store file (+ .fsi index)")
    b.add_argument("bam")
    b.add_argument("store")
    b.add_argument("--workers", type=int, default=20)
    b.add_argument("--chunk", type=int, default=1_000_000, help="genomic span per worker task")
    b.add_argument("--chroms", default=None, help="comma-separated subset")
    x = sub.add_parser("index", help="rebuild the sidecar index")
    x.add_argument("store")
    v = sub.add_parser("validate", help="compare random regions against the BAM")
    v.add_argument("bam")
    v.add_argument("store")
    v.add_argument("--n", type=int, default=300)
    v.add_argument("--seed", type=int, default=0)
    q = sub.add_parser("query", help="print fibers covering a region")
    q.add_argument("store")
    q.add_argument("region", help="chrom:start-end (0-based half-open)")
    q.add_argument("--min_frac", type=float, default=0.88)
    q.add_argument("--max_rows", type=int, default=5)
    i = sub.add_parser("info", help="store summary")
    i.add_argument("store")
    a = ap.parse_args(argv)

    if a.cmd == "build":
        from .builder import build
        build(a.bam, a.store, a.workers, a.chunk, a.chroms.split(",") if a.chroms else None)
    elif a.cmd == "index":
        from .store import FiberStore
        FiberStore(a.store, index=False).build_index()
    elif a.cmd == "validate":
        from .validation import validate
        sys.exit(0 if validate(a.bam, a.store, a.n, a.seed) else 1)
    elif a.cmd == "info":
        from .store import FiberStore
        fs = FiberStore(a.store)
        print(f"{len(fs)} fibers, {len(fs.chroms)} chromosomes, built {fs.meta.get('built')}")
        print(f"source: {fs.meta.get('source_bam')}")
        for c in fs.chroms:
            print(f"  {c}\t{len(fs.start[c])}")
    else:
        from .store import FiberStore
        c, s, e = parse_region(a.region)
        fs = FiberStore(a.store)
        t = time.perf_counter()
        out = fs.query(c, s, e, a.min_frac)
        dt = time.perf_counter() - t
        print(f"{len(out['row'])} fibers in {dt*1e3:.2f} ms")
        for i in range(min(a.max_rows, len(out["row"]))):
            print(f"  {fs.qname(out['movie'][i], out['zmw'][i])} {c}:{out['start'][i]}-{out['end'][i]} "
                  f"{'-' if out['reverse'][i] else '+'} m6A={len(out['m6a'][i])} "
                  f"5mC={len(out['c5m'][i])} nuc={len(out['nuc_s'][i])} msp={len(out['msp_s'][i])}")


if __name__ == "__main__":
    main()
