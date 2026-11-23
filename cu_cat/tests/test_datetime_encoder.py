import numpy as np
import pandas as pd

from cu_cat import DatetimeEncoder


def test_datetime_encoder_round_trip_from_list():
    """The docstring example: a list of dates encodes to year/month/day/hour."""
    X = [
        ["2022-10-15"],
        ["2021-12-25"],
        ["2020-05-18"],
        ["2019-10-15 12:00:00"],
    ]
    enc = DatetimeEncoder()
    enc.fit(X)
    out = np.asarray(enc.transform(X), dtype=float)
    expected = np.array(
        [
            [2022.0, 10.0, 15.0, 0.0],
            [2021.0, 12.0, 25.0, 0.0],
            [2020.0, 5.0, 18.0, 0.0],
            [2019.0, 10.0, 15.0, 12.0],
        ]
    )
    np.testing.assert_allclose(out, expected)


def test_datetime_encoder_missing_value_stays_missing():
    """A missing timestamp stays NaN. fillna(1500000) used to replace it."""
    X = pd.DataFrame(
        {"when": ["2020-01-01", None, "2021-06-15", "2022-07-20"]}
    )
    enc = DatetimeEncoder(extract_until="month")
    out = np.asarray(enc.fit_transform(X), dtype=float)
    assert out.shape == (4, 2)
    assert np.isnan(out[1]).all()
    assert not np.isnan(out[0]).any()
