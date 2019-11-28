from typing import Dict
from typing import Optional

from myrtlespeech.run.callbacks.callback import Callback


class RNNTTraining(Callback):
    """RNNT callback that **must** be used for RNNT training.

    The key functionality added here is to make the ground truth labels
    available to the :py:class:`Transducer` in the forward pass.
    """

    def __init__(self):
        super().__init__()

    def on_batch_begin(self, **kwargs) -> Optional[Dict]:
        """sets ``kwargs["last_input"] = (x, y)``"""
        x = kwargs.get("last_input")
        y = kwargs.get("last_target")
        assert x is not None, "kwargs['last_input'] must be set"
        assert y is not None, "kwargs['last_target'] must be set"

        kwargs["last_input"] = (x, y)

        return kwargs
