import numpy as np
from src.utils.rqe import RelativeQuantizationEncoder
from src.data.collate import pad_sequences_numpy


def test_rqe_shapes():
    T, D = 10, 6
    seq = np.random.randn(T, D).astype(np.float32)
    enc = RelativeQuantizationEncoder(num_bins=9)
    codes = enc.encode(seq)
    assert codes.shape == (T - 1, D)
    assert codes.dtype == np.int32


def test_pad_sequences_numpy():
    a = np.ones((5, 3), dtype=np.int32)
    b = np.zeros((3, 3), dtype=np.int32)
    batch, lengths = pad_sequences_numpy([a, b], pad_value=-1)
    assert batch.shape == (2, 5, 3)
    assert lengths.tolist() == [5, 3]
