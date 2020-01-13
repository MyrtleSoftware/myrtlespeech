"""
Utilities for preprocessing audio data.
"""
import random
from typing import Tuple

import torch

EPS = 1e-8


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

    def __call__(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns tuple of ``data`` and a tensor containing its length."""
        seq_len = data.size(self.length_dim)
        return data, torch.tensor([seq_len], requires_grad=False)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(length_dim={self.length_dim})"


class Standardize:
    """Normalises a tensor.

    Args:
        norm_type: Specifies the type of normalization:

            per_feature: Normalize each sample on a per-feature basis over
                all timesteps. In this case, input tensor must be of size
                ``channels=1, features, seq_length.``.

            all_features: Normalize each sample to have to have zero mean
                and one standard deviation. Tensor can be of arbitrary shape.


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

    def __init__(self, norm_type: str = "per_feature"):
        self.norm_type = norm_type

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Returns a tensor after subtracting mean and dividing by std.

        Args:
            tensor: A :py:class:`torch.Tensor` with the size specified in the
                initialization docstrings.
        """
        if self.norm_type == "all_features":
            return ((tensor - tensor.mean()) / tensor.std()).detach()
        elif self.norm_type == "per_feature":
            assert len(tensor.shape) == 3
            assert tensor.shape[0] == 1
            std_ = (tensor.std(dim=2) + EPS).unsqueeze(2)
            return (tensor - tensor.mean(dim=2).unsqueeze(2)) / std_
        else:
            raise ValueError(f"self.norm_type={self.norm_type} not recognised")

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


class SpecAugment:
    """`SpecAugment <https://arxiv.org/pdf/1904.08779.pdf>`_.

    Args:
        feature_mask: The maximum number of feature dimensions - typically
            frequencies - a single mask will zero. The actual number will be
            drawn from a uniform distribution from 0 to ``feature_mask`` each
            time SpecAugment is called. ``feature_mask`` is :math:`F` in the
            original paper.

        time_mask: The maximum number of time steps a single mask will zero.
            The actual number masked will be drawn from a uniform distribution
            from 0 to ``time_mask`` each time SpecAugment is called.
            ``time_mask`` is :math:`T` in the original paper.

        n_feature_masks: The number of feature masks to apply. :math:`m_F` in
            the original paper.

        n_time_masks: The number of time masks to apply. :math:`m_T` in the
            original paper.

    Raises:
        :py:class:`ValueError`: if any parameters are less than 0.
    """

    def __init__(
        self,
        feature_mask: int,
        time_mask: int,
        n_feature_masks: int = 1,
        n_time_masks: int = 1,
    ):
        if feature_mask < 0:
            raise ValueError(f"feature_mask={feature_mask} < 0")
        if time_mask < 0:
            raise ValueError(f"time_mask={time_mask} < 0")
        if n_feature_masks < 0:
            raise ValueError(f"n_feature_masks={n_feature_masks} < 0")
        if n_time_masks < 0:
            raise ValueError(f"n_time_masks={n_time_masks} < 0")

        self.feature_mask = feature_mask
        self.time_mask = time_mask
        self.n_feature_masks = n_feature_masks
        self.n_time_masks = n_time_masks

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Returns ``x`` after applying SpecAugment.

        Args:
            x: :py:class:`torch.Tensor` with size
                ``(channels, features, time steps)``.

        Returns:
            :py:class:`torch.Tensor` with size ``(channels, features, time
            steps)`` where some of the features and time steps may be set to 0.
        """
        _, n_features, n_time_steps = x.size()

        # mask features
        for _ in range(self.n_feature_masks):
            f_to_mask = random.randint(0, self.feature_mask)
            f_start = random.randint(0, max(0, n_features - f_to_mask))
            x[:, f_start : f_start + f_to_mask, :] = 0

        # mask time steps
        for _ in range(self.n_time_masks):
            t_to_mask = random.randint(0, self.time_mask)
            t_start = random.randint(0, max(0, n_time_steps - t_to_mask))
            x[:, :, t_start : t_start + t_to_mask] = 0

        return x

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(feature_mask={self.feature_mask},"
            + f" time_mask={self.time_mask},"
            + f" n_feature_masks={self.n_feature_masks},"
            + f" n_time_masks={self.n_time_masks})"
        )
