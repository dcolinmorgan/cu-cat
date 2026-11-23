# type: ignore

import collections
from typing import Any, Hashable
from inspect import getmodule
import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.utils import check_array
from ._dep_manager import deps
import subprocess as sp

cp = deps.cupy
cudf = deps.cudf
psutil = deps.psutil

# import cupy as cp

try:
    # Works for sklearn >= 1.0, < 1.7
    from sklearn.utils import parse_version  # noqa
except ImportError:
    # sklearn >= 1.7 keeps it private
    from sklearn.utils.fixes import parse_version  # noqa


class LRUDict:
    """dict with limited capacity

    Using LRU eviction avoids memorizing a full dataset"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def __getitem__(self, key: Hashable):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return -1

    def __setitem__(self, key: Hashable, value: Any):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def __contains__(self, key: Hashable):
        return key in self.cache


def _finite_check_kwargs():
    """Keyword ``check_array`` accepts for non-finite values.

    scikit-learn 1.6 renamed ``force_all_finite`` to ``ensure_all_finite``.
    1.8 removed the old name, so passing it on a current install is a TypeError.
    """
    if parse_version(sklearn_version) < parse_version("1.6"):
        return {"force_all_finite": False}
    return {"ensure_all_finite": False}


def _check_array(X):
    return check_array(
        X,
        dtype=None,
        ensure_2d=True,
        **_finite_check_kwargs(),
    )


def check_input(X) -> np.ndarray:
    """
    Check input with sklearn standards.
    Also converts X to a numpy array if not already.

    ``inspect.getmodule`` on an ndarray instance is ``None`` (the instance
    does not expose ``__module__``), so a module-name check never sees numpy
    and the frame branch then swallows ``AttributeError`` from ``.iloc``.
    """
    if isinstance(X, np.ndarray):
        # check_array(ensure_2d=True) rejects a 1-d array instead of
        # reshaping it. Promote a single feature so the validator runs.
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_ = _check_array(X)
        # If the array contains both NaNs and strings, convert to object type
        if X_.dtype.kind in {"U", "S"} and np.any(X_ == "nan"):
            return _check_array(np.array(X, dtype=object))
        return X_

    if cudf is not None and hasattr(X, "iloc"):
        for k in range(X.shape[1]):
            try:
                X.iloc[:, k] = cudf.to_numeric(X.iloc[:, k], downcast="float").to_cupy()
            except Exception:
                pass
    return X

def df_type(df):
    """
    Returns df type
    """

    try:  # if not cp:
        X = str(getmodule(df))
    except:  # if cp:
        X = str(cp.get_array_module(df))
    return X


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def get_sys_memory():
    # """
    # Get node total memory and memory usage
    # """
    # with open('/proc/meminfo', 'r') as mem:
    #     ret = {}
    #     tmp = 0
    #     for i in mem:
    #         sline = i.split()
    #         if str(sline[0]) == 'MemTotal:':
    #             ret['total'] = int(sline[1])
    #         elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
    #             tmp += int(sline[1])
    #     ret['free'] = tmp
    #     ret['used'] = int(ret['total']) - int(ret['free'])
    # return ret['free']
    psutil = deps.psutil
    stats = psutil.virtual_memory()  # returns a named tuple
    available = getattr(stats, 'available')
    return available


def _transform_one(transformer, X, y, weight, **fit_params):
    try:
        res = transformer.transform(X).to_output('cupy')
    except AttributeError:
        res = transformer.transform(X)
    except TypeError:
        res = transformer.transform(X.to_numpy())
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res
    return res * weight
