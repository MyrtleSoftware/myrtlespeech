from torch.optim.lr_scheduler import LambdaLR


class PolynomialLR(LambdaLR):
    """A polynomial decay learning rate scheduler.

    Follows the 'poly' decay rate described in `DeepLab: Semantic Image
    Segmentation with Deep Convolutional Nets, Atrous Convolution,
    and Fully Connected CRFs <https://arxiv.org/pdf/1606.00915.pdf>`_
    with a power of 2.

    Args:
        optimizer: An :py:class:`torch.optim.Optimizer`.

        total_steps: Total number of steps in training. This will be equal to
            ``steps_per_epoch * epochs``.

        min_lr_multiple: The minimum multiple of the lr.

        last_epoch: The index of the previous epoch.
    """

    def __init__(
        self, optimizer, total_steps, min_lr_multiple=0.01, last_epoch=-1
    ):
        self.total_steps = total_steps
        self.min_lr_multiple = min_lr_multiple
        super().__init__(
            optimizer, lr_lambda=self._poly_decay_fn, last_epoch=last_epoch
        )

    def _poly_decay_fn(self, step):
        res = ((self.total_steps - step) / self.total_steps) ** 2
        return max(res, self.min_lr_multiple)
