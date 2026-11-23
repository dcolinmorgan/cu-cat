import numpy as np

from cu_cat._utils import LRUDict, check_input


def test_check_input_validates_ndarray():
    """getmodule(ndarray) is None, so the old check never called check_array.

    ensure_2d promotes a 1-d vector. The pre-fix path returned it unchanged.
    """
    flat = np.array(["a", "b", "c"], dtype=object)
    out = check_input(flat)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 1)

    with_nan = np.array([[1.0, np.nan], [3.0, 4.0]])
    checked = check_input(with_nan)
    assert np.isnan(checked[0, 1])


def test_lrudict():
    dict_ = LRUDict(10)

    for x in range(15):
        dict_[x] = f"filled {x}"

    for x in range(5, 15):
        assert x in dict_
        assert dict_[x] == f"filled {x}"

    for x in range(5):
        assert x not in dict_
