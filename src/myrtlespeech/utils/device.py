import os

import torch


def get_device():
    r"""Returns device string.

    Device will be cuda iff :py:func:`torch.cuda.is_available` else cpu.
    GPU device index is the first of `os.environ['CUDA_VISIBLE_DEVICES']`
    """

    device = ""

    if torch.cuda.is_available:
        device = "cuda"
        indexes = os.environ.get("CUDA_VISIBLE_DEVICES")
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
