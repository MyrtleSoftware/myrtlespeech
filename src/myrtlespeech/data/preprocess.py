"""
Utilities for preprocessing audio data.
"""
from typing import Tuple

import torch


class AddSequenceLength:
    """Adds sequence length information to ``data``.

    Args:
        length_dim: Index of sequence length dimension.

    Example:
        >>> add_seq_len = AddSequenceLength(length_dim=0)
        >>> x = torch.rand([5, 10])
        >>> x_prime, x_len = add_seq_len(x)
        >>> bool(torch.all(x_prime == x))
        True
        >>> bool(x_len == torch.tensor(5))
        True
    """

    def __init__(self, length_dim=0):
        self.length_dim = length_dim

    def __call__(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns tuple of ``data`` and a tensor containing its length."""
        seq_len = data.size(self.length_dim)
        return data, torch.tensor([seq_len], requires_grad=False)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(length_dim={self.length_dim})"


class Standardize:
    """Standardize a tensor to have zero mean and one standard deviation.

    Example:
        >>> # Scale and shift standard normal distribution
        >>> x = 5*torch.empty(10000000).normal_() + 3
        >>> standardize = Standardize()
        >>> x_std = standardize(x)
        >>> bool(-0.001 <= x_std.mean() <= 0.001)
        True
        >>> bool(0.999 <= x_std.std() <= 1.001)
        True
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Returns a tensor after subtracting mean and dividing by std.

        Args:
           tensor: A :py:class:`torch.Tensor` with any number of dimensions.
        """
        return (tensor - tensor.mean()) / tensor.std()

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class AddContextFrames:
    """Adds context frames to each step in the input sequence.

    Args:
        n_context: The number of context frames (channels) to add to each frame
            in the input sequence.

    Example:
        >>> features = 3
        >>> seq_len = 5   # steps
        >>> x = torch.arange(features*seq_len).reshape(features, seq_len)
        >>> x = x.unsqueeze(0)  # add in channel=1 dimension
        >>> x
        tensor([[[ 0,  1,  2,  3,  4],
                 [ 5,  6,  7,  8,  9],
                 [10, 11, 12, 13, 14]]])
        >>> # compute expected result
        >>> exp = torch.tensor([
        ...     [[ 0,  0,  0,  1,  2],
        ...      [ 0,  0,  5,  6,  7],
        ...      [ 0,  0, 10, 11, 12]],
        ...     [[ 0,  0,  1,  2,  3],
        ...      [ 0,  5,  6,  7,  8],
        ...      [ 0, 10, 11, 12, 13]],
        ...     [[ 0,  1,  2,  3,  4],
        ...      [ 5,  6,  7,  8,  9],
        ...      [10, 11, 12, 13, 14]],
        ...     [[ 1,  2,  3,  4,  0],
        ...      [ 6,  7,  8,  9,  0],
        ...      [11, 12, 13, 14,  0]],
        ...     [[ 2,  3,  4,  0,  0],
        ...      [ 7,  8,  9,  0,  0],
        ...      [12, 13, 14,  0,  0]]
        ... ])
        >>> add_context_frames = AddContextFrames(n_context=2)
        >>> add_context_frames
        AddContextFrames(n_context=2)
        >>> bool(torch.all(add_context_frames(x) == exp))
        True
    """

    def __init__(self, n_context: int):
        self.n_context = n_context

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the :py:class:`torch.Tensor` after adding context frames.

        Args:
            x: :py:class:`torch.Tensor` with size ``(1, features, seq_len)``.

        Returns:
            A :py:class:`torch.Tensor` with size ``(2*n_context + 1, features,
            seq_len)``.
        """
        # Pad to ensure first and last n_context frames in original sequence
        # have at least n_context frames to their left and right respectively.
        assert x.size(0) == 1
        x = x.squeeze().T
        steps, features = x.shape
        padding = torch.zeros((self.n_context, features), dtype=x.dtype)
        x = torch.cat((padding, x, padding))

        window_size = self.n_context + 1 + self.n_context
        strides = x.stride()
        strided_x = torch.as_strided(
            x,
            # Shape of the new array.
            (steps, window_size, features),
            # Strides of the new array (bytes to step in each dim).
            (strides[0], strides[0], strides[1]),
        )

        return strided_x.clone().detach().permute(1, 2, 0)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(n_context={self.n_context})"
