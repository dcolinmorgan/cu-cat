import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from cu_cat import GapEncoder, TableVectorizer
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
    # Same seed, two fits. Multi-threaded BLAS is not bitwise stable: the
    # Python 3.10 CI image disagreed by about 5e-7. That is numerical noise,
    # not a different encoding.
    np.testing.assert_allclose(to_host(y), to_host(y2), rtol=1e-5, atol=1e-5)


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
    X = pd.DataFrame(np.array([["alice"], ["bob"]]))
    enc = GapEncoder(n_components=2, random_state=42)
    with pytest.raises(NotFittedError):
        enc.transform(X)

    # Check that it works after fit
    enc.fit(X)
    enc.transform(X)


def test_numpy_input_learns_columns_and_transforms():
    """ndarray columns are encoded, not rows, and transform does not need iloc."""
    X = np.array([
        ["alpha one", "RIGHT"],
        ["beta two", "RIGHT"],
        ["gamma three", "RIGHT"],
        ["delta four", "RIGHT"],
    ])
    enc = GapEncoder(n_components=2, max_iter=1, hashing=True, random_state=0)
    enc.fit(X)
    keys0 = {str(key) for key in enc.fitted_models_[0].H_dict_}
    keys1 = {str(key) for key in enc.fitted_models_[1].H_dict_}
    assert keys0 == {"alpha one", "beta two", "gamma three", "delta four"}
    assert keys1 == {"RIGHT"}
    transformed = to_host(enc.transform(X))
    assert transformed.shape == (X.shape[0], enc.n_components * X.shape[1])
    assert np.isfinite(transformed).all()


def test_small_sample():
    """Test that having n_samples < n_components raises an error"""
    X = np.array([["alice"], ["bob"]])
    enc = GapEncoder(n_components=3, random_state=42)
    with pytest.raises(ValueError, match="should be >= n_components"):
        enc.fit_transform(X)


def _midwest_region_sample() -> pd.DataFrame:
    """Twenty rows standing in for the OpenML midwest-survey region column.

    The two tests below only check that transform does not change the
    learned feature-name count. They used to download the survey; a short
    local column with repeated and held-out strings covers the same path.
    """
    regions = [
        "midwest", "the midwest", "Midwest", "mid west",
        "great lakes", "the great lakes", "Great Lakes",
        "east coast", "the east coast", "East Coast",
        "south", "the south", "deep south",
        "west coast", "the west coast", "pacific northwest",
        "plains", "the plains", "heartland", "corn belt",
    ]
    return pd.DataFrame(
        {"What_would_you_call_the_part_of_the_country_you_live_in_now": regions}
    )


def test_transform_shape():
    """Non-regression test for #188"""
    X_train, X_test = train_test_split(
        _midwest_region_sample(),
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
    X_train, X_test = train_test_split(
        _midwest_region_sample(),
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


def test_unique_lookup_reconstructs_input():
    """unq_X[lookup] must rebuild the input on every device.

    Guards the ordering contract between the unique values and the inverse
    index: they are consumed together by the batched update path, so a
    mismatch there silently encodes rows against the wrong topics.
    """
    from cu_cat._dep_manager import deps
    from cu_cat._gap_encoder import _unique_strings

    # deliberately unsorted, with repeats
    values = ["beta", "alpha", "gamma", "alpha", "beta", "beta", "delta"]
    engine = "cuml" if deps.cudf else "sklearn"

    unq_X, lookup = _unique_strings(pd.Series(values), engine, return_lookup=True)
    rebuilt = np.asarray(to_host(unq_X))[np.asarray(to_host(lookup))]

    assert list(rebuilt) == values


def test_transform_encodes_unseen_categories():
    """Strings absent from fit still get an encoding, via the tracked keys."""
    X_fit = pd.DataFrame({"c": ["alpha one", "beta two", "gamma three"] * 8})
    X_new = pd.DataFrame({"c": ["delta four", "alpha one", "epsilon five"]})

    enc = GapEncoder(n_components=3, max_iter=2, random_state=0, hashing=True)
    enc.fit(X_fit)
    out = to_host(enc.transform(X_new))

    assert out.shape == (3, 3)
    assert np.isfinite(out).all()
    # the two unseen strings must not collapse onto the same encoding
    assert not np.allclose(out[0], out[2])


def test_update_block_splits_on_out_of_memory(monkeypatch):
    """A block that exhausts memory is halved rather than failing the fit.

    The split must keep the activations returned for both halves. Reading
    ``unq_H`` back after the recursive call used to splice the stale first
    half onto an updated second half.
    """
    import cu_cat._gap_encoder as ge

    enc = GapEncoder(n_components=3, max_iter=2, random_state=0, hashing=True)
    enc.fit(generate_data(40, random_state=0))
    col = enc.fitted_models_[0]
    col.batch_size = 2

    words = pd.Series(["aaaa bbbb", "cccc dddd", "eeee ffff", "gggg hhhh",
                       "iiii jjjj", "kkkk llll", "mmmm nnnn", "oooo pppp"])
    words, _ = ge.make_safe_gpu_dataframes(words, None, col.engine)
    unq_V = col.ngrams_count_.transform(words)
    xp = ge.cp if (ge.cp is not None and "cupy" in ge.df_type(col.W_)) else np
    real = ge._multiplicative_update_h_smallfast
    saved = (col.W_.copy(), col.A_.copy(), col.B_.copy())

    def run_half(sl, H_init):
        H_new = real(
            unq_V[sl],
            col.W_,
            H_init,
            epsilon=1e-3,
            max_iter=col.max_iter_e_step,
            rescale_W=col.rescale_W,
            gamma_shape_prior=col.gamma_shape_prior,
            gamma_scale_prior=col.gamma_scale_prior,
        )
        ge._multiplicative_update_w_smallfast(
            unq_V[sl], col.W_, col.A_, col.B_, H_new, col.rescale_W, col.rho_,
        )
        return np.asarray(to_host(H_new))

    ones = np.ones((8, col.n_components))
    expected = np.vstack([
        run_half(slice(0, 4), ones[:4].copy()),
        run_half(slice(4, 8), ones[4:].copy()),
    ])
    col.W_, col.A_, col.B_ = saved

    widths = []

    def flaky(Vt, W, Ht, **kw):
        widths.append(Vt.shape[0])
        if Vt.shape[0] > 4:  # only the wide blocks "run out of memory"
            raise MemoryError("simulated")
        return real(Vt, W, Ht, **kw)

    monkeypatch.setattr(ge, "_multiplicative_update_h_smallfast", flaky)
    unq_H = xp.ones((8, col.n_components))
    out = col._update_block(unq_V, unq_H, slice(0, 8), 8)

    assert out.shape == (8, col.n_components)
    assert max(widths) == 8 and min(widths) <= 4  # tried wide, fell back narrow
    np.testing.assert_allclose(to_host(out), expected, rtol=1e-6, atol=1e-6)


def test_early_stopping_uses_cupy_norm_not_cp_module(monkeypatch):
    """Early stopping must not import a module named ``cp``.

    ``deps.cp`` is ``import cp``, which does not exist, so the CuPy norm
    never ran. The change is computed with ``deps.cupy`` when that imports.
    """
    import cu_cat._dep_manager as dm
    import cu_cat._gap_encoder as ge

    real_import = dm.importlib.import_module

    def guarded(name, *args, **kwargs):
        if name == "cp":
            raise AssertionError(
                "early stopping imported 'cp'; CuPy's module is 'cupy'"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(dm.importlib, "import_module", guarded)
    GapEncoder(
        n_components=2, max_iter=1, random_state=0, hashing=True
    ).fit(generate_data(12, random_state=0))

    calls = []

    class _Linalg:
        def norm(self, value):
            calls.append("norm")
            return np.linalg.norm(np.asarray(value))

    class _FakeCuPy:
        linalg = _Linalg()

        def multiply(self, left, right):
            calls.append("multiply")
            return np.multiply(left, right)

        def reciprocal(self, value):
            return np.reciprocal(np.asarray(value, dtype=float))

    monkeypatch.setattr(ge.deps, "cupy", _FakeCuPy(), raising=False)
    change = ge._relative_change(np.eye(2), np.eye(2) * 0.5)
    assert calls[:3] == ["norm", "norm", "multiply"]
    assert np.isfinite(change)


def test_chunked_transform_uses_max_iter_e_step(monkeypatch):
    """Chunked transform uses the same E-step cap as the in-memory path."""
    import cu_cat._gap_encoder as ge

    enc = GapEncoder(
        n_components=2, max_iter=1, max_iter_e_step=7, random_state=0, hashing=True,
    )
    enc.fit(generate_data(12, random_state=0))
    col = enc.fitted_models_[0]
    seen = []
    real = ge._multiplicative_update_h_smallfast

    def spy(Vt, W, Ht, **kwargs):
        seen.append(kwargs.get("max_iter"))
        return real(Vt, W, Ht, **kwargs)

    monkeypatch.setattr(ge, "_multiplicative_update_h_smallfast", spy)
    monkeypatch.setattr(col, "_plan_updates", lambda sh, sw: (False, 2))
    col.transform(pd.Series(["aaaa bbbb", "cccc dddd", "eeee ffff", "gggg hhhh"]))
    assert seen
    assert set(seen) == {7}


def test_resolve_engine_sklearn_when_cudf_imported(monkeypatch):
    """An explicit sklearn engine is the CPU path even if cuDF imported."""
    import cu_cat._gap_encoder as ge

    monkeypatch.setattr(ge.deps, "cudf", object(), raising=False)
    monkeypatch.setattr(ge.deps, "cupy", object(), raising=False)
    monkeypatch.setattr(ge.deps, "cupyx", object(), raising=False)
    monkeypatch.setattr(ge.deps, "cuml", object(), raising=False)
    assert ge.resolve_engine("sklearn")[0] == "sklearn"


def test_resolve_engine_falls_back_when_cupy_missing(monkeypatch):
    import cu_cat._gap_encoder as ge

    monkeypatch.setattr(ge.deps, "cudf", object(), raising=False)
    monkeypatch.setattr(ge.deps, "cupy", None, raising=False)
    monkeypatch.setattr(ge.deps, "cupyx", None, raising=False)
    monkeypatch.setattr(ge.deps, "cuml", None, raising=False)
    assert ge.resolve_engine("auto")[0] == "sklearn"
    assert ge.resolve_engine("cuml")[0] == "sklearn"


def test_update_block_reports_levers_when_it_cannot_shrink():
    """At the floor the error names what the user can actually change."""
    import cu_cat._gap_encoder as ge

    enc = GapEncoder(n_components=3, max_iter=2, random_state=0,
                     hashing=True, hashing_n_features=256)
    enc.fit(generate_data(40, random_state=0))
    col = enc.fitted_models_[0]
    col.batch_size = 64  # floor above the block we ask for

    class Boom:
        shape = (8, 4)
        def __getitem__(self, _):
            raise MemoryError("simulated")

    with pytest.raises(MemoryError, match="hashing_n_features"):
        col._update_block(Boom(), np.ones((8, 3)), slice(0, 8), 8)
