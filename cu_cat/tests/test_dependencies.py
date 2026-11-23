import pytest

from cu_cat._dep_manager import DepManager


def test_runtime_dependencies_import():
    """The install set has to import together.

    Weekly CI failed once pyarrow 26 (needs NumPy 2) was resolved next to
    the dirty-cat extra's scikit-learn<=1.3.2 pin (which keeps NumPy 1.26).
    """
    import numpy
    import pyarrow
    import scipy
    import sklearn

    import cu_cat

    assert hasattr(pyarrow, "StringScalar")
    assert int(numpy.__version__.split(".")[0]) >= 2
    assert sklearn.__version__
    assert scipy.__version__
    assert cu_cat.GapEncoder is not None


def test_dep_manager_missing_module_is_none():
    deps = DepManager()
    assert deps.there_is_no_such_module_cu_cat_zzz is None


def test_dep_manager_broken_import_is_not_swallowed(monkeypatch):
    import cu_cat._dep_manager as dm

    real = dm.importlib.import_module

    def boom(name, *args, **kwargs):
        if name == "broken_wheel_mod":
            raise ImportError("binary wheel mismatch")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(dm.importlib, "import_module", boom)
    deps = DepManager()
    with pytest.raises(ImportError, match="binary wheel"):
        _ = deps.broken_wheel_mod
