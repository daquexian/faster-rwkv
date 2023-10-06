from fr_python import _Tensor, Model, Sampler, Tokenizer

import numpy as np
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("fasterrwkv")
except PackageNotFoundError:
    __version__ = "unknown version"

_Tensor.numpy = lambda self: np.array(self, copy=False)

# _run returns a fr Tensor, but we don't want to expose that to the user.
Model.__call__ = lambda self, *args, **kwargs: self._run(*args, **kwargs).numpy()

# _run returns a fr Tensor, but we don't want to expose that to the user.
Sampler.sample = lambda self, logits, *args, **kwargs: self._sample(_Tensor(logits), *args, **kwargs)
Sampler.__call__ = Sampler.sample
