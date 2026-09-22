# type: ignore

import collections
from typing import Any, Hashable
from inspect import getmodule
import numpy as np
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


def check_input(X) -> np.ndarray:
    """
    Check input with sklearn standards.
    Also converts X to a numpy array if not already.
    """
    # TODO check for weird type of input to pass scikit learn tests
    #  without messing with the original type too much
    if 'numpy' in str(getmodule(X)):
        X_ = check_array(
            X,
            dtype=None,
            ensure_2d=True,
            force_all_finite=False,
        )
        # If the array contains both NaNs and strings, convert to object type
        if X_.dtype.kind in {"U", "S"}:  # contains strings
            if np.any(X_ == "nan"):  # missing value converted to string
                return check_array(
                    np.array(X, dtype=object),  # had been using cp here, but not necessary
                    dtype=None,
                    ensure_2d=True,
                    force_all_finite=False,
                )
    if 'numpy' not in str(getmodule(X)):
        for k in range(X.shape[1]):
            try:
                X.iloc[:,k] = cudf.to_numeric(X.iloc[:,k], downcast='float').to_cupy()
            except:
                pass
        X_ = X
    
    return X_

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
