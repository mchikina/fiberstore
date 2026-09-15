import pytest

pysam = pytest.importorskip("pysam")

from fiberstore.builder import build            # noqa: E402
from fiberstore.store import FiberStore       # noqa: E402
from fiberstore.testing import synthetic_bam  # noqa: E402


@pytest.fixture(scope="session")
def bam(tmp_path_factory):
    d = tmp_path_factory.mktemp("bam")
    path = d / "syn.bam"
    truth, reference = synthetic_bam(path, n_reads=700, seed=1)
    return path, truth, reference


@pytest.fixture(scope="session")
def store(bam, tmp_path_factory):
    path, truth, _ = bam
    out = tmp_path_factory.mktemp("store") / "fs"
    # small chunks so every chromosome has several parts, most with a partial row group
    build(str(path), str(out), workers=2, chunk=5_000, log=None)
    return FiberStore(str(out)), truth
