from typing import Union

import torch
from apex import amp
from myrtlespeech.model.seq_to_seq import SeqToSeq
from myrtlespeech.run.callbacks.callback import ModelCallback


class ClipGradNorm(ModelCallback):
    """Clips gradient norm of the model's parameters.

    If :py:class:`.MixedPrecision` is used then this must appear later in the
    list of callbacks (i.e. have a higher index) as the
    :py:class:`.MixedPrecision` callback needs to first rescale the gradients.

    Uses :py:func:`torch.nn.utils.clip_grad_norm_` internally.

    Args:
        model: See :py:class:`ModelCallback`.
        max_norm: See :py:func:`torch.nn.utils.clip_grad_norm_`.
        norm_type: See :py:func:`torch.nn.utils.clip_grad_norm_`.
    """

    def __init__(
        self,
        model: SeqToSeq,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2,
    ):
        super().__init__(model)
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_backward_end(self, **kwargs):
        torch.nn.utils.clip_grad_norm_(
            parameters=amp.master_params(self.model.optim),
            max_norm=self.max_norm,
            norm_type=self.norm_type,
        )
