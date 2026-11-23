from typing import Dict, List, Literal, Optional,no_type_check
from warnings import warn

import numpy as _numpy
import pandas as pd

from ._dep_manager import deps

cudf = deps.cudf
if cudf is None:
    cudf = deps.pandas
    np = deps.numpy
else:
    np = deps.cupy

from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._utils import parse_version  # type: ignore

# Required for ignoring lines too long in the docstrings
# flake8: noqa: E501

WORD_TO_ALIAS: Dict[str, str] = {
    "year": "Y",
    "month": "M",
    "day": "D",
    "hour": "h",
    "minute": "min",
    "second": "S",
    "millisecond": "ms",
    "microsecond": "us",
    "nanosecond": "N",
}
TIME_LEVELS: List[str] = list(WORD_TO_ALIAS.keys())
AcceptedTimeValues = Literal[
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
]


class DatetimeEncoder(TransformerMixin, BaseEstimator):
    """
    Transforms each datetime column into several numeric columns for temporal features (e.g year, month, day...).

    Constant extracted features are dropped; for instance, if the year is
    always the same in a feature, the extracted "year" column won't be added.
    If the dates are timezone aware, all the features extracted will correspond
    to the provided timezone.

    Parameters
    ----------
    extract_until : AcceptedTimeValues, default="hour"
        Extract up to this granularity.
        If all features have not been extracted, add the "total_time" feature,
        which contains the time to epoch (in seconds).
        For instance, if you specify "day", only "year", "month", "day" and
        "total_time" features will be created.
    add_day_of_the_week : bool, default=False
        Add day of the week feature (if day is extracted).
        This is a numerical feature from 0 (Monday) to 6 (Sunday).

    Attributes
    ----------
    n_features_in_: int
        Number of features in the data seen during fit.
    n_features_out_ : int
        Number of features of the transformed data.
    features_per_column_ : mapping of int to list of str
        Dictionary mapping the index of the original columns
        to the list of features extracted for each column.
    col_names_ : None or list of str
        List of the names of the features of the input data,
        if input data was a pandas DataFrame, otherwise None.

    See Also
    --------
    :class:`~cu_cat.GapEncoder` :
        Encodes dirty categories (strings) by constructing latent topics with continuous encoding.
    :class:`~cu_cat.MinHashEncoder` :
        Encode string columns as a numeric array with the minhash method.
    :class:`~cu_cat.SimilarityEncoder` :
        Encode string columns as a numeric array with n-gram string similarity.

    Examples
    --------
    >>> enc = DatetimeEncoder()

    Let's encode the following dates:

    >>> X = [['2022-10-15'], ['2021-12-25'], ['2020-05-18'], ['2019-10-15 12:00:00']]

    >>> enc.fit(X)
    DatetimeEncoder()

    The encoder will output a transformed array
    with four columns (year, month, day and hour):

    >>> enc.transform(X)
    array([[2022.,   10.,   15.,    0.],
           [2021.,   12.,   25.,    0.],
           [2020.,    5.,   18.,    0.],
           [2019.,   10.,   15.,   12.]])
    """

    n_features_in_: int
    n_features_out_: int
    features_per_column_: Dict[int, List[str]]
    col_names_: Optional[List[str]]

    def __init__(
        self,
        extract_until: AcceptedTimeValues = "hour",
        add_day_of_the_week: bool = False,
    ):
        self.extract_until = extract_until
        self.add_day_of_the_week = add_day_of_the_week

    def _more_tags(self):
        """
        Used internally by sklearn to ease the estimator checks.
        """
        return {"X_types": ["categorical"]}

    def _validate_keywords(self):
        if self.extract_until not in TIME_LEVELS:
            raise ValueError(
                f'"extract_until" should be one of {TIME_LEVELS}, '
                f"got {self.extract_until}. "
            )

    @staticmethod
    def _coerce_frame(X):
        """DataFrame with one column per datetime feature.

        A list has no ``.shape``. pandas is used when the value is not already
        a cuDF frame, including on the CPU path where ``cudf`` is aliased to
        pandas and ``Series.to_pandas`` does not exist.
        """
        gpu = deps.cudf
        if gpu is not None and isinstance(X, gpu.DataFrame):
            return X
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, pd.Series):
            return X.to_frame()
        return pd.DataFrame(X)

    @staticmethod
    def _as_datetime_index(date_series):
        if hasattr(date_series, "to_pandas") and type(date_series).__module__.startswith("cudf"):
            host = date_series.to_pandas()
        elif isinstance(date_series, pd.Series):
            host = date_series
        else:
            host = pd.Series(date_series)
        # format="mixed" keeps both "2022-10-15" and "2019-10-15 12:00:00"
        # in one column. The default parser infers one format and turns the
        # other into NaT.
        return pd.DatetimeIndex(
            pd.to_datetime(host, errors="coerce", format="mixed")
        )

    @staticmethod
    @no_type_check
    def _extract_from_date(date_series, feature: str):
        idx = DatetimeEncoder._as_datetime_index(date_series)
        if feature == "millisecond":
            # pandas DatetimeIndex exposes microsecond, not millisecond.
            values = _numpy.asarray(idx.microsecond, dtype=float) // 1000.0
        elif feature == "total_time":
            stamped = idx.tz_convert("UTC") if idx.tz is not None else idx
            values = stamped.asi8.astype(float) / 1e9
        else:
            values = _numpy.asarray(getattr(idx, feature), dtype=float)
        values = _numpy.asarray(values, dtype=float).copy()
        values[_numpy.asarray(idx.isna())] = _numpy.nan
        return values

    def fit(self, X, y=None) -> "DatetimeEncoder":
        """Fit the instance to X.

        In practice, just stores which extracted features are not constant.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data where each column is a datetime feature.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        :class:`~cu_cat.DatetimeEncoder`
            Fitted :class:`~cu_cat.DatetimeEncoder` instance (self).
        """
        self._validate_keywords()
        # Columns to extract for each column,
        # before taking into account constant columns
        self._to_extract = TIME_LEVELS[: TIME_LEVELS.index(self.extract_until) + 1]
        if self.add_day_of_the_week:
            self._to_extract.append("dayofweek")
        named = isinstance(X, pd.DataFrame) or (
            deps.cudf is not None and isinstance(X, deps.cudf.DataFrame)
        )
        if named:
            self.col_names_ = list(X.columns)
        else:
            self.col_names_ = None
        X = self._coerce_frame(X)
        # Features to extract for each column, after removing constant features
        self.features_per_column_ = {}
        for i in range(X.shape[1]):
            self.features_per_column_[i] = []
        # Check which columns are constant
        for i in range(X.shape[1]):
            for feature in self._to_extract:
                extracted = self._extract_from_date(X.iloc[:, i], feature)
                if _numpy.nanstd(extracted) > 0:
                    self.features_per_column_[i].append(feature)
            # If some date features have not been extracted, then add the
            # "total_time" feature, which contains the full time to epoch
            # remainder = (
            #     cudf.to_datetime(X.iloc[:, i])
            #     - cudf.to_datetime(
            #         cudf.DatetimeIndex(X.iloc[:, i]).floor(WORD_TO_ALIAS[self.extract_until])
            #     )
            # ).seconds.to_numpy()
            # if np.nanstd(remainder) > 0:
            #     self.features_per_column_[i].append("total_time")

        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = sum(
            len(features) for features in self.features_per_column_.values()
        )

        return self
    
    @no_type_check
    def transform(self, X, y=None) -> np.ndarray:
        """Transform X by replacing each datetime column with corresponding numerical features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to transform, where each column is a datetime feature.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        :obj:`~numpy.ndarray`, shape (n_samples, n_features_out_)
            Transformed input.
        """
        check_is_fitted(
            self,
            attributes=["n_features_in_", "n_features_out_", "features_per_column_"],
        )
        X = self._coerce_frame(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"The number of features in the input data ({X.shape[1]}) "
                "does not match the number of features "
                f"seen during fit ({self.n_features_in_}). "
            )
        # Missing datetimes stay missing. Filling them with an integer before
        # extracting parts (the previous CPU path) destroyed the timestamps,
        # and a pandas Series has no ``to_pandas``.
        if self.n_features_out_ == 0:
            return _numpy.empty((len(X), 0), dtype=float)
        columns = []
        for i in range(X.shape[1]):
            for feature in self.features_per_column_[i]:
                columns.append(self._extract_from_date(X.iloc[:, i], feature))
        return _numpy.column_stack(columns)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Returns clean feature names with format "<column_name>_<new_feature>"
        if the original data has column names, otherwise with format
        "<column_index>_<new_feature>" where `<new_feature>` is one of
        ["year", "month", "day", "hour", "minute", "second", "millisecond",
        "microsecond", "nanosecond", "dayofweek"].
        """
        feature_names = []
        names = input_features if input_features is not None else self.col_names_
        for i in self.features_per_column_.keys():
            prefix = str(i) if names is None else str(names[i])
            for feature in self.features_per_column_[i]:
                feature_names.append(f"{prefix}_{feature}")
        return feature_names

    def get_feature_names(self, input_features=None) -> List[str]:
        """
        Ensures compatibility with sklearn < 1.0, and returns the output of
        get_feature_names_out.
        """
        if parse_version(sklearn_version) >= parse_version("1.0"):
            warn(
                "Following the changes in scikit-learn 1.0, "
                "get_feature_names is deprecated. "
                "Use get_feature_names_out instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.get_feature_names_out()
