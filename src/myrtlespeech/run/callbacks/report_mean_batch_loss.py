from typing import Optional

from myrtlespeech.run.callbacks.callback import Callback


class ReportMeanBatchLoss(Callback):
    """Reports the mean batch loss for an epoch.

    Example:
        >>> # imports
        >>> import torch
        >>> from myrtlespeech.run.callbacks.callback import CallbackHandler
        >>>
        >>> # initialize
        >>> report_loss = ReportMeanBatchLoss()
        >>> cb_handler = CallbackHandler(callbacks=[report_loss])
        >>>
        >>> # simulate training for 1 epoch containing 2 batches
        >>> cb_handler.on_train_begin(epochs=1)
        >>> _ = cb_handler.train(mode=True)
        >>> _ = cb_handler.on_epoch_begin()
        >>> _ = cb_handler.on_backward_begin(loss=torch.tensor([0.0]))
        >>> _ = cb_handler.on_batch_end()
        >>> _ = cb_handler.on_backward_begin(loss=torch.tensor([10.0]))
        >>> _ = cb_handler.on_batch_end()
        >>> _ = cb_handler.on_epoch_end()
        >>>
        >>> cb_handler.state_dict["reports"]["ReportMeanBatchLoss"]
        5.0
    """

    def __init__(self):
        super().__init__()

    def _reset(self, **kwargs) -> None:
        kwargs["reports"][self.__class__.__name__] = None
        self.loss: Optional[float] = None

    def on_train_begin(self, **kwargs) -> None:
        """Sets ``kwargs["reports"]["ReportMeanBatchLoss"] = None``."""
        self._reset(**kwargs)

    def on_epoch_begin(self, **kwargs) -> None:
        """Sets ``kwargs["reports"]["ReportMeanBatchLoss"] = None``."""
        self._reset(**kwargs)

    def on_backward_begin(self, **kwargs) -> None:
        """Adds ``kwargs["last_loss"].item()`` to ``self.loss``."""
        if self.loss is None:
            self.loss = kwargs["last_loss"].item()
        else:
            self.loss += kwargs["last_loss"].item()

    def on_epoch_end(self, **kwargs) -> None:
        """Sets ``kwargs["reports"]["ReportMeanBatchLoss"]`` to mean loss."""
        self.loss = 0.0 if self.loss is None else self.loss
        kwargs["reports"][self.__class__.__name__] = self.loss / float(
            kwargs["epoch_batches"]
        )
