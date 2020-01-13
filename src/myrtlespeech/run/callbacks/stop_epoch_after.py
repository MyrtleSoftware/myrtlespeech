from typing import Dict
from typing import Optional

from myrtlespeech.run.callbacks.callback import Callback


class StopEpochAfter(Callback):
    """Stops an epoch after ``epoch_minibatches`` have been processed.

    This is a small class for convenience.

    Example:
        >>> # imports
        >>> from myrtlespeech.run.callbacks.callback import CallbackHandler
        >>>
        >>> # initialize
        >>> stop_epoch_after = StopEpochAfter(epoch_minibatches=5)
        >>> cb_handler = CallbackHandler(callbacks=[stop_epoch_after])
        >>>
        >>> # simulate training for 1 epoch containing 2 batches
        >>> cb_handler.on_train_begin(epochs=1)
        >>> _ = cb_handler.train(mode=True)
        >>>
        >>> for i in range(5):
        ...     stop_epoch = cb_handler.on_batch_end()
        ...     print(i, stop_epoch)
        0 False
        1 False
        2 False
        3 False
        4 True
    """

    def __init__(self, epoch_minibatches: int = 1):
        self.epoch_minibatches = epoch_minibatches

    def on_batch_end(self, **kwargs) -> Optional[Dict]:
        """Returns ``{'stop_epoch': True}`` after ``epoch_minibatches``."""
        if kwargs["epoch_minibatches"] + 1 >= self.epoch_minibatches:
            return {"stop_epoch": True}
        return None  # keep mypy happy
