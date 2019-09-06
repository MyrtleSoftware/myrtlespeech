from typing import Callable

import torch


class Lambda(torch.nn.Module):
    """Apply a :py:class:`torch.nn.Module` to apply an arbitrary function.

    Args:
        lambda_fn: The function to apply when :py:meth:`Lambda.forward` is
            called.

    Example:

        >>> scale = Lambda(lambda_fn=lambda x: x*2.0)
        >>> scale(torch.tensor([1.0]))
        tensor([2.])
    """

    def __init__(self, lambda_fn: Callable):
        super().__init__()
        self.lambda_fn = lambda_fn

    def forward(self, x):
        """Returns ``lambda_fn(x)``."""
        return self.lambda_fn(x)
