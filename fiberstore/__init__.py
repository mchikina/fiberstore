"""fiberstore: compact, region-queryable per-fiber store built from a Fiber-seq BAM.

Reading a store needs numpy + pyarrow only::

    from fiberstore import FiberStore
    fs = FiberStore("treg.parquet")
    t = fs.query("chr1", 100_000_000, 100_001_000, min_frac=0.88)

Building (``fiberstore build BAM STORE.parquet``) additionally needs pysam.
"""
from .builder import build
from .codec import ROW_GROUP, SCHEMA, dec_len, dec_pos, enc_ints, enc_pos
from .store import FiberStore, read_meta
from .validation import validate

__version__ = "0.2.0"
__all__ = ["FiberStore", "SCHEMA", "ROW_GROUP", "enc_ints", "enc_pos", "dec_pos", "dec_len",
           "build", "validate", "read_meta"]

