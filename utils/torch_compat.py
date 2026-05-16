"""PyTorch>=2.6 defaults ``torch.load(..., weights_only=True)``; legacy checkpoints need full unpickle."""

from __future__ import annotations

from typing import Any

import torch


def torch_load_compat(path: Any, map_location: Any = "cpu", **kwargs: Any):
    """Load trusted local checkpoints / embedding dicts (same as pre-2.6 ``torch.load``)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, map_location=map_location, **kwargs)
