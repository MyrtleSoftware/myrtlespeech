from torch.optim.lr_scheduler import LambdaLR


class ConstantLR(LambdaLR):
    """A constant learning rate scheduler."""

    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(
            optimizer, lr_lambda=lambda _: 1, last_epoch=last_epoch
        )
