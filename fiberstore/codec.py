"""Byte encoding of per-fiber integer arrays and the Parquet schema.

Encoding (uint8 escape scheme, decodable with one numpy cumsum): a sequence of
non-negative integers d_i is written byte-wise as ESC repeated d_i // ESC times
followed by d_i % ESC.  Decoding:
    cs = cumsum(bytes); values = cs[bytes != ESC]      (cumulative -> positions)
    or take differences of those values                (-> lengths, see dec_len)
"""
import numpy as np
import pyarrow as pa

FORMAT = 2                 # bump when the file layout changes
ROW_GROUP = 256            # fibers per Parquet row group (the fetch granularity)
MAX_READ_LEN = 1 << 17     # index search slack; must exceed the longest fiber
ESC = 255
META_KEY = b"fiberstore"   # key of the JSON blob in the Parquet schema metadata
INDEX_SUFFIX = ".fsi"      # sidecar index: <store>.parquet.fsi (like .bam.bai)

SCHEMA = pa.schema([
    ("chrom", pa.string()),
    ("start", pa.uint32()), ("end", pa.uint32()), ("reverse", pa.bool_()),
    ("rg", pa.uint8()), ("movie", pa.uint8()), ("zmw", pa.uint32()), ("qlen", pa.uint32()),
    ("np", pa.uint8()), ("rq", pa.float32()), ("nm", pa.uint16()),
    ("m6a", pa.binary()), ("c5m", pa.binary()), ("c5m_ml", pa.binary()),
    ("nuc_s", pa.binary()), ("nuc_l", pa.binary()), ("msp_s", pa.binary()), ("msp_l", pa.binary()),
])
PART_SCHEMA = pa.schema([f for f in SCHEMA if f.name != "chrom"])   # per-chunk temp files
STATS_COLS = ["chrom", "start", "end"]          # the only columns given Parquet statistics
BIN_COLS = ["m6a", "c5m", "c5m_ml", "nuc_s", "nuc_l", "msp_s", "msp_l"]
POS_COLS = ("m6a", "c5m", "nuc_s", "msp_s")     # delta-coded positions relative to `start`
LEN_COLS = ("nuc_l", "msp_l")                   # escape-coded lengths
RAW_COLS = ("c5m_ml",)                          # one byte per element, as is


def enc_ints(d):
    """Escape-encode a non-negative int array to bytes."""
    d = np.asarray(d, dtype=np.int64)
    if d.size == 0:
        return b""
    if d.min() < 0:
        raise ValueError("enc_ints: negative value")
    q, r = np.divmod(d, ESC)
    if q.max() == 0:
        return r.astype(np.uint8).tobytes()
    out = np.full(int(q.sum() + d.size), ESC, np.uint8)
    out[np.cumsum(q + 1) - 1] = r
    return out.tobytes()


def dec_ints(buf):
    """Inverse of enc_ints."""
    b = np.frombuffer(buf, np.uint8)
    if b.size == 0:
        return np.zeros(0, np.int64)
    cs = np.cumsum(b.astype(np.int64))
    v = cs[b != ESC]
    return np.diff(v, prepend=0)


def enc_pos(pos, origin):
    """Sorted positions -> delta-coded bytes relative to origin (pos[0] >= origin)."""
    pos = np.asarray(pos, dtype=np.int64)
    if pos.size == 0:
        return b""
    return enc_ints(np.diff(pos, prepend=origin))


def dec_pos(buf, origin):
    """Inverse of enc_pos."""
    b = np.frombuffer(buf, np.uint8)
    if b.size == 0:
        return np.zeros(0, np.int64)
    return origin + np.cumsum(b.astype(np.int64))[b != ESC]


dec_len = dec_ints
