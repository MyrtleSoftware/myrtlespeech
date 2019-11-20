from typing import Collection
from typing import Optional

from myrtlespeech.model.seq_to_seq import SeqToSeq
from myrtlespeech.run.callbacks.callback import Callback
from myrtlespeech.run.callbacks.callback import CallbackHandler
from myrtlespeech.run.train import run_stage
from torch.utils.data import DataLoader


def eval(
    seq_to_seq: SeqToSeq,
    eval_loader: DataLoader,
    callbacks: Optional[Collection[Callback]] = None,
) -> None:
    r"""Eval ``seq_to_seq`` for ``eval_loader`` iters.

    Args:
        seq_to_seq: A :py:class:`.SeqToSeq` model.

        eval_loader: A :py:class:`torch.utils.data.DataLoader` for
            the validation data.

        callbacks: An optional collection of :py:class:`.Callback`\s.
    """
    is_training = False
    cb_handler = CallbackHandler(callbacks, is_training)
    cb_handler.on_train_begin(epochs=1)

    run_stage(seq_to_seq, cb_handler, eval_loader, is_training=is_training)
