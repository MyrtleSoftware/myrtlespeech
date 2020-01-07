from torch.optim.lr_scheduler import LambdaLR


class PolynomialLR(LambdaLR):
    """A polynomial decay learning rate scheduler."""

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
