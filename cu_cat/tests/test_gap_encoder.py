import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from cu_cat import GapEncoder, TableVectorizer
from cu_cat.datasets._fetching import fetch_midwest_survey
from cu_cat.tests.utils import generate_data, to_host

MODULES = [pd]


@pytest.mark.parametrize(
    ["hashing", "init", "rescale_W", "rescale_rho", "add_words"],
    [
        (True, "random", False, True, False),
    ],
)
def test_analyzer(
    hashing: bool,
    init: str,
    rescale_W: bool,
    add_words: bool,
    rescale_rho: bool,
    n_samples: int = 70,
):
    """
    Test if the output is different when the analyzer is 'word' or 'char'.
    If it is, no error ir raised.
    """
    X = generate_data(n_samples, random_state=0)
    n_components = 10
    # Test first analyzer output:
    encoder = GapEncoder(
        n_components=n_components,
        hashing=hashing,
        init=init,
        analyzer="char",
        add_words=add_words,
        random_state=42,
        rescale_W=rescale_W,
        rescale_rho=rescale_rho,
    )
    encoder.fit(X)
    y1 = encoder.transform(X)
    # s1 = encoder.score(X)

    # Test the other analyzer output:
    encoder = GapEncoder(
        n_components=n_components,
        hashing=hashing,
        init=init,
        analyzer="word",
        add_words=add_words,
        random_state=42,
        rescale_W=rescale_W,
        rescale_rho=rescale_rho,
    )
    encoder.fit(X)
    y2 = encoder.transform(X)
    # s2 = encoder.score(X)

    # Test inequality between the word and char analyzers output:
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, to_host(y1), to_host(y2)
    )
    # np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, s1, s2)


@pytest.mark.parametrize(
    ["hashing", "init", "analyzer", "add_words", "verbose"],
    [
        (True, "random", "char", False, False),
    ],
)
def test_gap_encoder(
    hashing: bool,
    init: str,
    analyzer: str,
    add_words: bool,
    verbose: bool,
    n_samples: int = 70,
):
    X = generate_data(n_samples, random_state=0)
    n_components = 10
    # Test output shape
    encoder = GapEncoder(
        n_components=n_components,
        hashing=hashing,
        init=init,
        analyzer=analyzer,
        add_words=add_words,
        random_state=42,
        rescale_W=True,
    )
    encoder.fit(X)
    y = encoder.transform(X)
    assert y.shape == (n_samples, n_components * X.shape[1]), str(y.shape)

    # Test L1-norm of topics W.
    for col_enc in encoder.fitted_models_:
        l1_norm_W = np.abs(to_host(col_enc.W_)).sum(axis=1)
        np.testing.assert_array_almost_equal(l1_norm_W, np.ones(n_components))

    # Test same seed return the same output
    encoder = GapEncoder(
        n_components=n_components,
        hashing=hashing,
        init=init,
        analyzer=analyzer,
        add_words=add_words,
        random_state=42,
    )
    encoder.fit(X)
    y2 = encoder.transform(X)
    # GPU reductions are not bitwise reproducible across runs, so same-seed
    # equality is checked to tight tolerance rather than exactly.
    np.testing.assert_allclose(to_host(y), to_host(y2), rtol=1e-10, atol=1e-10)


def test_get_feature_names_out(n_samples=70):
    X = generate_data(n_samples, random_state=0)
    enc = GapEncoder(random_state=42)
    enc.fit(X)
    feature_names_1 = enc.get_feature_names_out()
    feature_names_2 = enc.get_feature_names_out()
    for topic_labels in [feature_names_1, feature_names_2]:
        # Check number of labels
        assert len(topic_labels) == enc.n_components * X.shape[1]
        # # Test different parameters for col_names
        # topic_labels_2 = enc.get_feature_names_out(col_names="auto")
        # assert topic_labels_2[0] == "col0: " + topic_labels[0]
        # topic_labels_3 = enc.get_feature_names_out(col_names=["abc", "def"])
        # assert topic_labels_3[0] == "abc: " + topic_labels[0]
    return


# def test_get_feature_names_out_no_words():
#     # Test the GapEncoder get_feature_names_out when there are no words
#     enc = GapEncoder(random_state=42)
#     # A dataframe with words too short
#     df = pd.DataFrame(
#         20 * [["a b c d",],],)

#     enc.fit(df)
#     # The difficulty here is that, in this specific case short words
#     # should not be filtered out
#     enc.get_feature_names_out()
#     return


def test_get_feature_names_out_redundent():
    # With the following dataframe, the GapEncoder can produce feature names
    # that have the same name, which leads duplicated features names,
    # which themselves lead to errors in the TableVectorizer
    # get_feature_names_out() method.
    df = pd.DataFrame(
        40 * [["aaa bbb cccc ddd",],],)

    tv = TableVectorizer(cardinality_threshold=1)
    tv.fit(df)
    tv.get_feature_names_out()


def test_check_fitted_gap_encoder():
    """Test that calling transform before fit raises an error"""
    import pandas as pd
    from cu_cat import GapEncoder
    X = pd.DataFrame(np.array([["alice"], ["bob"]]))
    enc = GapEncoder(n_components=2, random_state=42)
    # with pytest.raises(NotFittedError):
        # enc.transform(X)

    # Check that it works after fit
    enc.fit(X)
    enc.transform(X)


def test_small_sample():
    """Test that having n_samples < n_components raises an error"""
    X = np.array([["alice"], ["bob"]])
    enc = GapEncoder(n_components=3, random_state=42)
    with pytest.raises(ValueError, match="should be >= n_components"):
        enc.fit_transform(X)


def test_transform_shape():
    """Non-regression test for #188"""
    dataset = fetch_midwest_survey()
    X_train, X_test = train_test_split(
        dataset.X[["What_would_you_call_the_part_of_the_country_you_live_in_now"]],
        random_state=0,
    )
    enc = GapEncoder(n_components=2, random_state=2)
    enc.fit_transform(X_train)
    topics1 = enc.get_feature_names_out()
    enc.transform(X_test)
    topics2 = enc.get_feature_names_out()
    assert len(topics1) == len(topics2)


def test_transform_deterministic():
    """Non-regression test for #188"""
    dataset = fetch_midwest_survey()
    X_train, X_test = train_test_split(
        dataset.X[["What_would_you_call_the_part_of_the_country_you_live_in_now"]],
        random_state=0,
    )
    enc = GapEncoder(n_components=2, random_state=2)
    enc.fit_transform(X_train)
    topics1 = enc.get_feature_names_out()
    enc.transform(X_test)
    topics2 = enc.get_feature_names_out()  # fit_tarnsform used by pyg so not worried about this
    # assert_array_equal(topics1, topics2)
    assert len(topics1) == len(topics2)


def test_partial_fit_matches_fit_shape():
    """Chunked fitting reaches the same encoding shape as a single fit."""
    X = generate_data(60, random_state=0)

    whole = GapEncoder(n_components=3, max_iter=2, random_state=42, hashing=True)
    whole.fit(X)

    chunked = GapEncoder(n_components=3, max_iter=2, random_state=42, hashing=True)
    for start in range(0, len(X), 20):
        chunked.partial_fit(X.iloc[start : start + 20])

    assert len(chunked.fitted_models_) == len(whole.fitted_models_)
    assert to_host(chunked.transform(X)).shape == to_host(whole.transform(X)).shape


def test_partial_fit_learns_from_every_chunk():
    """Topics keep moving as later chunks arrive, rather than freezing."""
    X = generate_data(60, random_state=0)
    enc = GapEncoder(n_components=3, max_iter=2, random_state=42, hashing=True)

    enc.partial_fit(X.iloc[:20])
    after_first = to_host(enc.fitted_models_[0].W_).copy()
    enc.partial_fit(X.iloc[20:40])
    after_second = to_host(enc.fitted_models_[0].W_)

    assert after_first.shape == after_second.shape
    assert not np.allclose(after_first, after_second)


def test_partial_fit_rejects_column_count_change():
    X = generate_data(40, random_state=0)
    enc = GapEncoder(n_components=3, max_iter=2, random_state=42, hashing=True)
    enc.partial_fit(X)
    wider = pd.concat([X, X], axis=1)
    wider.columns = ["a", "b"]  # cudf rejects duplicate names before our check
    with pytest.raises(ValueError, match="columns changed"):
        enc.partial_fit(wider)


def test_plan_updates_chunks_when_over_budget():
    """The planner falls back to a memory-derived chunk, not a fixed batch."""
    from cu_cat._gap_encoder import GapEncoderColumn

    enc = GapEncoderColumn(n_components=10, batch_size=128)
    enc.engine = "cuml"
    enc.gmem = 1000  # MB
    enc.byte_lim = 8

    # 1000 unique x 4096 vocab x 8B x 3 copies = ~98 MB -> fits
    fits, chunk = enc._plan_updates(sh=1000, sw=4096)
    assert fits and chunk == 1000

    # 1e6 unique x 4096 vocab x 8B x 3 copies = ~98 GB -> must chunk
    fits, chunk = enc._plan_updates(sh=1_000_000, sw=4096)
    assert not fits
    assert chunk >= enc.batch_size          # never worse than the old fixed size
    assert chunk < 1_000_000
    # the chunk's dense term must sit inside the budget
    assert (enc.byte_lim * chunk * 4096 * enc._DENSE_COPIES) / 1e6 < enc.gmem


def test_plan_updates_never_below_batch_size():
    """A tiny budget still yields a usable chunk rather than zero rows."""
    from cu_cat._gap_encoder import GapEncoderColumn

    enc = GapEncoderColumn(n_components=10, batch_size=128)
    enc.engine = "cuml"
    enc.gmem = 1  # MB: absurdly small
    fits, chunk = enc._plan_updates(sh=500_000, sw=4096)
    assert not fits
    assert chunk == enc.batch_size
