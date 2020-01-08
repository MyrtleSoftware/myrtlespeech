from myrtlespeech.lr_scheduler.base import _LambdaLR


class ConstantLR(_LambdaLR):
    """A constant learning rate scheduler."""

    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(
            optimizer, lr_lambda=lambda _: 1, last_epoch=last_epoch
        )
