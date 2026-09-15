import numpy as np
import pytest

from fiberstore.cli import main
from fiberstore.codec import ROW_GROUP
from fiberstore.validation import validate


def _rows_by_key(fs, chrom):
    n = len(fs.start[chrom])
    t = fs.fetch(chrom, np.arange(n))
    return {(fs.qname(m, z), int(s)): i for i, (m, z, s) in enumerate(zip(t["movie"], t["zmw"], t["start"]))}, t


def test_primary_only_and_complete(store):
    fs, truth = store
    assert len(fs) == len(truth)
    for c in fs.chroms:
        keys, _ = _rows_by_key(fs, c)
        assert set(keys) == {k for k, v in truth.items() if v["chrom"] == c}
        assert np.all(np.diff(fs.start[c].astype(np.int64)) >= 0)


def test_every_field_matches_truth(store):
    fs, truth = store
    for c in fs.chroms:
        keys, t = _rows_by_key(fs, c)
        for k, i in keys.items():
            tr = truth[k]
            assert t["end"][i] == tr["end"] and bool(t["reverse"][i]) == tr["reverse"]
            assert t["qlen"][i] == tr["qlen"] and t["rg"][i] == tr["rg"] and t["movie"][i] == tr["movie"]
            assert t["np"][i] == tr["np"] and t["nm"][i] == tr["nm"]
            assert abs(float(t["rq"][i]) - tr["rq"]) < 1e-6
            assert np.array_equal(t["m6a"][i], tr["m6a"]), k
            assert np.array_equal(t["c5m"][i], tr["c5m"]), k
            assert np.array_equal(t["c5m_ml"][i], tr["c5m_ml"]), k
            for kind in ("nuc", "msp"):
                got = list(zip(t[f"{kind}_s"][i].tolist(), (t[f"{kind}_s"][i] + t[f"{kind}_l"][i]).tolist()))
                assert got == tr[kind], (k, kind)


def test_row_groups_uniform(store):
    fs, _ = store
    for c in fs.chroms:
        pf, offsets = fs._file(c)
        sizes = np.diff(offsets)
        if len(fs.start[c]) > ROW_GROUP:
            assert pf.metadata.num_row_groups >= 2
        assert np.all(sizes[:-1] == ROW_GROUP) and 0 < sizes[-1] <= ROW_GROUP


def test_select_matches_bruteforce(store):
    fs, _ = store
    rng = np.random.default_rng(0)
    for c in fs.chroms:
        st, en = fs.start[c].astype(np.int64), fs.end[c].astype(np.int64)
        L = int(en.max())
        for _ in range(200):
            s = int(rng.integers(0, L))
            e = s + int(rng.integers(1, 3000))
            frac = float(rng.choice([0.0, 0.5, 0.88, 1.0]))
            ov = np.minimum(en, e) - np.maximum(st, s)
            need = max(1, frac * (e - s))
            assert np.array_equal(fs.select(c, s, e, frac), np.nonzero(ov >= need)[0])
        assert fs.count(c, [0, 100], [50, 200]).tolist() == [len(fs.select(c, 0, 50)), len(fs.select(c, 100, 200))]


def test_fetch_scattered_rows_and_columns(store):
    fs, _ = store
    c = fs.chroms[0]
    n = len(fs.start[c])
    rows = np.array([0, 1, ROW_GROUP - 1, ROW_GROUP, n - 1, 2 * ROW_GROUP + 3])
    rows = np.unique(rows[rows < n])
    full = fs.fetch(c, np.arange(n))
    part = fs.fetch(c, rows, columns=["zmw", "m6a", "nuc_l"])
    assert set(part) == {"zmw", "m6a", "nuc_l"}
    assert np.array_equal(part["zmw"], full["zmw"][rows])
    for j, r in enumerate(rows):
        assert np.array_equal(part["m6a"][j], full["m6a"][r])
        assert np.array_equal(part["nuc_l"][j], full["nuc_l"][r])
    raw = fs.fetch(c, rows, columns=["m6a"], decode=False)
    assert all(isinstance(b, bytes) for b in raw["m6a"])
    empty = fs.fetch(c, [], columns=["zmw", "m6a"])
    assert len(empty["zmw"]) == 0 and empty["m6a"] == []


def test_query_and_intervals(store):
    fs, _ = store
    c = fs.chroms[0]
    q = fs.query(c, 10_000, 11_000, min_frac=0.88, columns=["start", "end"])
    assert len(q["row"]) > 0
    assert np.all(np.minimum(q["end"], 11_000) - np.maximum(q["start"], 10_000) >= 880)
    s, e = fs.intervals(c, q["row"], "nuc")
    for a, b in zip(s, e):
        assert np.all(b > a)


def test_validate_against_bam(bam, store):
    path, _, _ = bam
    fs, _ = store
    assert validate(str(path), fs.path, n=40, region_len=500, min_start=0, log=None)


def test_cli_query_and_info(store, capsys):
    fs, _ = store
    c = fs.chroms[0]
    main(["query", fs.path, f"{c}:10000-11000", "--min_frac", "0.5"])
    out = capsys.readouterr().out
    assert "fibers in" in out and "m6A=" in out
    main(["info", fs.path])
    assert f"{len(fs)} fibers" in capsys.readouterr().out
