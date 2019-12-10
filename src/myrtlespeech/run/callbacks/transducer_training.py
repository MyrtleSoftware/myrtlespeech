from typing import Dict
from typing import Optional

import torch
from myrtlespeech.run.callbacks.callback import Callback


class TransducerForward(Callback):
    """Callback that **must** be used for Transducer forward pass.

    This makes the ground truth labels available to the
    :py:class:`Transducer` during the forward pass.

    .. note::

        User **should not** initialise this callback. If the
        :py:class:`CallbackHandler` class is used, this will be handled
        interneally.
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

        # Check that this Callback has not been used twice
        if not isinstance(x[0], torch.Tensor):
            raise ValueError(
                f"`kwargs['last_input']` should be a Tuple of torch.Tensors "
                f"but is a Tuple with first element of type={type(x[0])}. "
                "Have you used the TransducerForward Callback more than once? "
                "\n\nNote: the user **should not** pass the TransducerForward "
                "callback to the fit function."
            )

        return kwargs
