import numpy as np
import pytest

from fiberstore.codec import ESC, dec_ints, dec_len, dec_pos, enc_ints, enc_pos


@pytest.mark.parametrize("vals", [
    [], [0], [1, 2, 3], [254], [255], [256], [510], [511], [1000, 0, 70000, 3],
    list(range(0, 2000, 7)),
])
def test_ints_roundtrip(vals):
    assert dec_ints(enc_ints(vals)).tolist() == vals


def test_ints_random():
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = rng.integers(0, 3000, rng.integers(0, 200))
        assert np.array_equal(dec_ints(enc_ints(v)), v)


def test_escape_layout():
    assert enc_ints([255]) == bytes([ESC, 0])
    assert enc_ints([256]) == bytes([ESC, 1])
    assert enc_ints([510, 1]) == bytes([ESC, ESC, 0, 1])


def test_pos_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        origin = int(rng.integers(0, 10**8))
        pos = origin + np.sort(rng.choice(200_000, rng.integers(0, 500), replace=False))
        assert np.array_equal(dec_pos(enc_pos(pos, origin), origin), pos)
    assert dec_pos(b"", 5).size == 0


def test_pos_first_equals_origin():
    assert dec_pos(enc_pos([10, 11], 10), 10).tolist() == [10, 11]


def test_negative_rejected():
    with pytest.raises(ValueError):
        enc_ints([-1])


def test_dec_len_alias():
    assert dec_len is dec_ints
