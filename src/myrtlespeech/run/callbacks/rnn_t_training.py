from typing import Dict
from typing import Optional

from myrtlespeech.run.callbacks.callback import Callback


class RNNTTraining(Callback):
    """RNNT callback that *must* be used for all RNNT training.

    The key functionality added here is to make the labels available
    to the rnnt model in the forward pass.

    """

    def __init__(self):
        super().__init__()

    def on_batch_begin(self, **kwargs) -> Optional[Dict]:
        x = kwargs.get("last_input")
        y = kwargs.get("last_target")
        assert x is not None, "kwargs['last_input'] must be set"
        assert y is not None, "kwargs['last_target'] must be set"

        x_0, x_1 = x  # (audio inputs, audio lens)
        y_0, y_1 = y  # (target 'inputs', target lens)

        # re-arrange rnnt inputs - see myrtlespeech.model.rnn_t.RNNT
        # forward() docstring for reasoning:

        del kwargs["last_input"]

        kwargs["last_input"] = ((x_0, y_0), (x_1, y_1))

        del x, y, x_0, y_0, x_1, y_1

        return kwargs
