import random

import torch


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
