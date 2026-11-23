from __future__ import annotations
from time import time
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from sklearn.utils.validation import check_is_fitted

from sklearn import __version__ as sklearn_version

from cu_cat._datetime_encoder import DatetimeEncoder
from cu_cat._gap_encoder import GapEncoder
from cu_cat._table_vectorizer import TableVectorizer
from cu_cat._dep_manager import deps
from cu_cat._utils import parse_version
try:
    # Installed-but-incompatible: dirty-cat 0.4.2 imports
    # sklearn.utils.fixes._parse_version, which scikit-learn 1.9 removed.
    # That is an ImportError, not a missing module, and DepManager lets it
    # surface. The comparisons below stay skipped.
    dirty_cat = deps.dirty_cat
except ImportError:
    dirty_cat = None


MSG_PANDAS_DEPRECATED_WARNING = "Skip deprecation warning"


def _dirty_cat_unusable() -> bool:
    """dirty-cat 0.4.2 does not run on the stack pip now resolves.

    Checked against the installed dirty-cat 0.4.2 sources:

    * Importing it on scikit-learn 1.9 raises ImportError:
      ``sklearn.utils.fixes._parse_version`` is gone (it already left
      ``sklearn.utils.parse_version`` in 1.7).
    * ``DatetimeEncoder.fit`` floors with the pandas alias ``"H"``
      (``WORD_TO_ALIAS["hour"]``). pandas 2.2 warns; pandas 3.0 rejects it.
    * ``check_input`` still passes ``force_all_finite``, which scikit-learn
      1.8 removed. Python 3.11 and 3.12 resolve scikit-learn 1.8+.

    The four comparisons also download remote CSVs and, except
    ``test_small_news``, assert that cu-cat is faster. That is a timing
    check, so leaving them skipped is the CI-stable result. Unskipping
    would need a maintained dirty-cat (skrub) and offline fixtures.
    """
    if dirty_cat is None:
        return True
    pandas_breaks = parse_version(pd.__version__) >= parse_version("2.2")
    sklearn_breaks = parse_version(sklearn_version) >= parse_version("1.8")
    return pandas_breaks or sklearn_breaks


requires_dirty_cat = pytest.mark.skipif(
    _dirty_cat_unusable(),
    reason=(
        "dirty-cat 0.4.2 does not import on scikit-learn>=1.7 "
        "(_parse_version removed), floors datetimes with pandas alias 'H', "
        "and passes force_all_finite"
    ),
)


def check_same_transformers(
    expected_transformers: dict, actual_transformers: list
) -> None:
    # Construct the dict from the actual transformers
    actual_transformers_dict = {name: cols for name, trans,
                                cols in actual_transformers}
    assert actual_transformers_dict == expected_transformers


def type_equality(expected_type, actual_type) -> bool:
    """
    Checks that the expected type is equal to the actual type,
    assuming object and str types are equivalent
    (considered as categorical by the TableVectorizer).
    """
    if (isinstance(expected_type, object) or isinstance(expected_type, str)) and (
        isinstance(actual_type, object) or isinstance(actual_type, str)
    ):
        return True
    else:
        return expected_type == actual_type


def _get_clean_dataframe() -> pd.DataFrame:
    """
    Creates a simple DataFrame with various types of data,
    and without missing values.
    """
    return pd.DataFrame(
        {
            "int": pd.Series([15, 56, 63, 12, 44], dtype="int"),
            "float": pd.Series([5.2, 2.4, 6.2, 10.45, 9.0], dtype="float"),
            "str1": pd.Series(
                ["public", "private", "private", "private", "public"], dtype="string"
            ),
            "str2": pd.Series(
                ["officer", "manager", "lawyer", "chef", "teacher"], dtype="string"
            ),
            "cat1": pd.Series(["yes", "yes", "no", "yes", "no"], dtype="category"),
            "cat2": pd.Series(
                ["20K+", "40K+", "60K+", "30K+", "50K+"], dtype="category"
            ),
        }
    )


def _get_dirty_dataframe(categorical_dtype="object") -> pd.DataFrame:
    """
    Creates a simple DataFrame with some missing values.
    We'll use different types of missing values (np.nan, pd.NA, None)
    to test the robustness of the vectorizer.
    """
    return pd.DataFrame(
        {
            "int": pd.Series([15, 56, pd.NA, 12, 44], dtype="Int64"),
            "float": pd.Series([5.2, 2.4, 6.2, 10.45, np.nan], dtype="Float64"),
            "str1": pd.Series(
                ["public", np.nan, "private", "private", "public"],
                dtype=categorical_dtype,
            ),
            "str2": pd.Series(
                ["officer", "manager", None, "chef", "teacher"],
                dtype=categorical_dtype,
            ),
            "cat1": pd.Series(
                [np.nan, "yes", "no", "yes", "no"], dtype=categorical_dtype
            ),
            "cat2": pd.Series(
                ["20K+", "40K+", "60K+", "30K+", np.nan], dtype=categorical_dtype
            ),
        }
    )


def _get_mixed_types_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "int_str": ["1", "2", 3, "3", 5],
            "float_str": ["1.0", pd.NA, 3.0, "3.0", 5.0],
            "int_float": [1, 2, 3.0, 3, 5.0],
            "bool_str": ["True", False, True, "False", "True"],
        }
    )


def _get_mixed_types_array() -> np.ndarray:
    return np.array(
        [
            ["1", "2", 3, "3", 5],
            ["1.0", np.nan, 3.0, "3.0", 5.0],
            [1, 2, 3.0, 3, 5.0],
            ["True", False, True, "False", "True"],
        ]
    ).T


def _get_list_of_lists() -> list:
    return _get_numpy_array().tolist()


def _test_possibilities(X) -> None:
    """
    Do a bunch of tests with the TableVectorizer.
    We take some expected transformers results as argument. They're usually
    lists or dictionaries.
    """
    # Test with low cardinality and a StandardScaler for the numeric columns
    vectorizer_base = TableVectorizer(
        cardinality_threshold=4,
        # we must have n_samples = 5 >= n_components
        high_card_cat_transformer=GapEncoder(n_components=2),
        numerical_transformer=StandardScaler(),
    )
    # Warning: order-dependant
    expected_transformers_df = {
        "numeric": ["int", "float"],
        "low_cardinality": ["str1", "cat1"],
        "high_cardinality": ["str2", "cat2"],
    }
    vectorizer_base.fit_transform(X)
    check_same_transformers(expected_transformers_df, vectorizer_base.transformers_)

    # Test with higher cardinality threshold and no numeric transformer
    expected_transformers_2 = {
        "low_cardinality": ["str1", "str2", "cat1", "cat2"],
        "numeric": ["int", "float"],
    }
    vectorizer_default = TableVectorizer()  # Using default values
    vectorizer_default.fit_transform(X)
    check_same_transformers(expected_transformers_2, vectorizer_default.transformers_)

    # Test with single column dataframe
    expected_transformers_series = {
        "low_cardinality": ["cat1"],
    }
    vectorizer_base.fit_transform(X[["cat1"]])
    check_same_transformers(expected_transformers_series, vectorizer_base.transformers_)

    # Test casting values
    vectorizer_cast = TableVectorizer(
        cardinality_threshold=4,
        # we must have n_samples = 5 >= n_components
        high_card_cat_transformer=GapEncoder(n_components=2),
        numerical_transformer=StandardScaler(),
    )
    X_str = X.astype("object")
    # With pandas
    expected_transformers_plain = {
        "high_cardinality": ["str2", "cat2"],
        "low_cardinality": ["str1", "cat1"],
        "numeric": ["int", "float"],
    }
    vectorizer_cast.fit_transform(X_str)
    check_same_transformers(expected_transformers_plain, vectorizer_cast.transformers_)
    # With numpy
    expected_transformers_np_cast = {
        "numeric": [0, 1],
        "low_cardinality": [2, 4],
        "high_cardinality": [3, 5],
    }
    vectorizer_cast.fit_transform(X_str.to_numpy())
    check_same_transformers(
        expected_transformers_np_cast, vectorizer_cast.transformers_
    )


def test_with_clean_data() -> None:
    """
    Defines the expected returns of the vectorizer in different settings,
    and runs the tests with a clean dataset.
    """
    _test_possibilities(_get_clean_dataframe())


def test_with_dirty_data() -> None:
    """
    Defines the expected returns of the vectorizer in different settings,
    and runs the tests with a dataset containing missing values.
    """
    _test_possibilities(_get_dirty_dataframe(categorical_dtype="object"))
    _test_possibilities(_get_dirty_dataframe(categorical_dtype="category"))


def test_get_feature_names_out() -> None:
    X = _get_clean_dataframe()

    vec_w_pass = TableVectorizer(remainder="passthrough")
    vec_w_pass.fit(X)

    # Order matters: numeric columns, then each low-cardinality column's
    # categories in sorted order. Comparing lengths hid the padded names.
    expected_feature_names = [
        "int",
        "float",
        "str1_private",
        "str1_public",
        "str2_chef",
        "str2_lawyer",
        "str2_manager",
        "str2_officer",
        "str2_teacher",
        "cat1_no",
        "cat1_yes",
        "cat2_20K+",
        "cat2_30K+",
        "cat2_40K+",
        "cat2_50K+",
        "cat2_60K+",
    ]
    assert list(vec_w_pass.get_feature_names_out()) == expected_feature_names

    vec_w_drop = TableVectorizer(remainder="drop")
    vec_w_drop.fit(X)
    assert list(vec_w_drop.get_feature_names_out()) == expected_feature_names


def test_short_object_categories_are_not_zero_padded() -> None:
    """yes/no must stay yes/no. rjust(4) used to emit 00no and 0yes."""
    X = pd.DataFrame(
        {
            "yesno": ["yes", "no", "yes", "no", "yes"],
            "num": [1, 2, 3, 4, 5],
        }
    )
    vec = TableVectorizer()
    vec.fit(X)
    names = list(vec.get_feature_names_out())
    assert "yesno_yes" in names
    assert "yesno_no" in names
    assert not any(name.startswith("yesno_0") for name in names)


def test_datetime_passthrough_without_cudf() -> None:
    """CPU fit_transform must not call cudf.concat for a datetime column."""
    X = pd.DataFrame(
        {
            "when": pd.to_datetime(
                ["2020-01-02", "2021-06-15", "2022-03-03", "2019-12-31"]
            ),
            "city": ["paris", "lyon", "paris", "nice"],
        }
    )
    vec = TableVectorizer(cardinality_threshold=10)
    out = vec.fit_transform(X)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == 4
    assert out.shape[1] >= 4
    assert any(pd.api.types.is_datetime64_any_dtype(out[col]) for col in out.columns)
    assert "datetime" not in [name for name, _, _ in vec.transformers_]


def test_datetime_transformer_argument_is_honored() -> None:
    X = pd.DataFrame(
        {
            "when": pd.to_datetime(
                ["2020-01-02", "2021-06-15", "2022-03-03", "2019-12-31"]
            ),
            "city": ["paris", "lyon", "paris", "nice"],
        }
    )
    vec = TableVectorizer(
        cardinality_threshold=10,
        datetime_transformer=DatetimeEncoder(extract_until="year"),
    )
    vec.fit(X)
    assert any(name == "datetime" for name, _, _ in vec.transformers_)
    names = [str(name) for name in vec.get_feature_names_out()]
    assert any(name.endswith("_year") for name in names)
    transformed = np.asarray(vec.transform(X), dtype=float)
    assert transformed.shape[0] == 4
    present = set(np.round(transformed[np.isfinite(transformed)]).astype(int))
    assert {2019, 2020, 2021, 2022} <= present


def test_fit() -> None:
    # Simply checks sklearn's `check_is_fitted` function raises an error if
    # the TableVectorizer is instantiated but not fitted.
    # See GH#193
    table_vec = TableVectorizer()
    with pytest.raises(NotFittedError):
        assert check_is_fitted(table_vec)


def test_check_fitted_table_vectorizer() -> None:
    """Test that calling transform before fit raises an error"""
    X = _get_clean_dataframe()
    tv = TableVectorizer()
    with pytest.raises(NotFittedError):
        tv.transform(X)

    # Test that calling transform after fit works
    tv.fit(X)
    tv.transform(X)


@requires_dirty_cat
def test_HN():
    # from cu_cat import TableVectorizer
    # import pandas as pd
    askHN = pd.read_csv('https://storage.googleapis.com/cohere-assets/blog/text-clustering/data/askhn3k_df.csv', index_col=0)
    table_vec = TableVectorizer()
    t = time()
    aa = table_vec.fit_transform((askHN))
    ct = time() - t
    # if deps.dirty_cat:
    t = time()
    bb = dirty_cat.TableVectorizer().fit_transform(askHN)
    dt = time() - t
    assert aa.shape[0] == bb.shape[0]
    assert ct < dt
    # else:
    #     assert aa.shape[0] == askHN.shape[0]


@requires_dirty_cat
def test_red_team():
    df = pd.read_csv('https://gist.githubusercontent.com/silkspace/c7b50d0c03dc59f63c48d68d696958ff/raw/31d918267f86f8252d42d2e9597ba6fc03fcdac2/redteam_50k.csv', index_col=0)
    red_team = pd.read_csv('https://gist.githubusercontent.com/silkspace/5cf5a94b9ac4b4ffe38904f20d93edb1/raw/888dabd86f88ea747cf9ff5f6c44725e21536465/redteam_labels.csv', index_col=0)
    df['feats'] = df.src_computer + ' ' + df.dst_computer + ' ' + df.auth_type + ' ' + df.logontype
    df['feats2'] = df.src_computer + ' ' + df.dst_computer
    ndf = df.drop_duplicates(subset=['feats'])
    tdf = pd.concat([red_team.reset_index(), ndf.reset_index()])
    tdf['node'] = range(len(tdf))
    table_vec = TableVectorizer()
    t = time()
    aa = table_vec.fit_transform((tdf))
    ct = time() - t
    # if deps.dirty_cat:
    t = time()
    bb = dirty_cat.TableVectorizer().fit_transform(tdf)
    dt = time() - t
    assert aa.shape[0] == bb.shape[0]
    assert ct < dt
    # else:
    #     assert aa.shape[0] == tdf.shape[0]


@requires_dirty_cat
def test_malware():
    edf = pd.read_csv('https://gist.githubusercontent.com/silkspace/33bde3e69ae24fee1298a66d1e00b467/raw/dc66bd6f1687270be7098f94b3929d6a055b4438/malware_bots.csv', index_col=0)
    T = edf.Label.apply(lambda x: True if 'Botnet' in x else False)
    bot = edf[T]
    nbot = edf[~T]
    print(f'Botnet abundance: {100*len(bot)/len(edf):0.2f}%')  # so botnet traffic makes up a tiny fraction of total

    # let's balance the dataset in a 10-1 ratio, for speed and demonstrative purposes
    negs = nbot.sample(10*len(bot))
    edf = pd.concat([bot, negs])  # top part of arrays are bot traffic, then all non-bot traffic
    edf = edf.drop_duplicates()
    table_vec = TableVectorizer()
    t = time()
    aa = table_vec.fit_transform((edf))
    ct = time() - t
    # if deps.dirty_cat:
    t = time()
    bb = dirty_cat.TableVectorizer().fit_transform(edf)
    dt = time() - t
    assert aa.shape[0] == bb.shape[0]
    assert ct < dt
    # else:
        # assert aa.shape[0] == edf.shape[0]


@requires_dirty_cat
def test_small_news():
    from sklearn.datasets import fetch_20newsgroups
    n_samples = 1000

    news, _ = fetch_20newsgroups(
        shuffle=True,
        random_state=1,
        remove=("headers", "footers", "quotes"),
        return_X_y=True,
    )

    news = news[:n_samples]
    news = pd.DataFrame(news)
    table_vec = TableVectorizer()
    t = time()
    aa = table_vec.fit_transform((news))
    ct = time() - t
    # if deps.dirty_cat:
    t = time()
    bb = dirty_cat.TableVectorizer().fit_transform(news)
    dt = time() - t
    assert aa.shape[0] == bb.shape[0]
    # assert ct < dt  # only GPU is faster
    # else:
        # assert aa.shape[0] == news.shape[0]


# def test_large_news():
#     from sklearn.datasets import fetch_20newsgroups
#     n_samples = 3000

#     news, _ = fetch_20newsgroups(
#         shuffle=True,
#         random_state=1,
#         remove=("headers", "footers", "quotes"),
#         return_X_y=True,
#     )

#     news = news[:n_samples]
#     news = pd.DataFrame(news)
#     table_vec = TableVectorizer()
#     t = time()
#     aa = table_vec.fit_transform((news))
#     ct = time() - t
#     # if deps.dirty_cat:
#     t = time()
#     bb = dirty_cat.TableVectorizer().fit_transform(news)
#     dt = time() - t
#     assert aa.shape[0] == bb.shape[0]
#     assert ct < dt  # only GPU is fatser, but also this gets killed by github

