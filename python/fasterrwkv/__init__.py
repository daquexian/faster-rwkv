from fr_python import _Tensor, Model, Sampler, Tokenizer, layernorm

import numpy as np
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("fasterrwkv")
except PackageNotFoundError:
    __version__ = "unknown version"

def tensor_numpy(self):
    self = self.cpu()
    arr = np.array(self, copy=False)
    if arr.dtype == np.int16:
        arr = arr.view(np.float16)
    return arr

_Tensor.numpy = tensor_numpy

# _run returns a fr Tensor, but we don't want to expose that to the user.
Model.__call__ = lambda self, *args, **kwargs: self._run(*args, **kwargs).numpy()

# _run returns a fr Tensor, but we don't want to expose that to the user.
Sampler.sample = lambda self, logits, *args, **kwargs: self._sample(_Tensor(logits), *args, **kwargs)
Sampler.__call__ = Sampler.sample
