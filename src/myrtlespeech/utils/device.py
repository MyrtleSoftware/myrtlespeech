import os
from typing import Optional

import torch


def get_device(use_cuda: Optional[bool] = None) -> str:
    r"""Returns device string.

    If GPU is used, device index is the first of
    `os.environ['CUDA_VISIBLE_DEVICES']`.

    Args:
        use_cuda: Optional boolean. If True, forces use of gpu and if False,
            forces use of cpu. If None, cuda will be used if
            :py:func:`torch.cuda.is_available`.

    Returns:
        String representation of torch device.
    """
    if use_cuda is not None:
        torch.cuda.is_available()

    device = ""
    indexes = os.environ.get("CUDA_VISIBLE_DEVICES")

    if indexes != "" and use_cuda:
        device = "cuda"
        if indexes:
            if len(indexes) > 1:
                idx = indexes.split(",")[0]
            else:
                idx = indexes
        else:
            idx = "0"
        assert (
            idx.isnumeric()
        ), "os.environ['CUDA_VISIBLE_DEVICES'] must be numeric"
        device += ":" + idx
    else:
        device = "cpu"
    return device
