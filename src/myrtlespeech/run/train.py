from contextlib import ExitStack
from typing import Collection
from typing import Optional

import torch
from myrtlespeech.model.seq_to_seq import SeqToSeq
from myrtlespeech.run.callback import Callback
from myrtlespeech.run.callback import CallbackHandler
from myrtlespeech.stage import Stage
from torch.utils.data import DataLoader


def fit(
    seq_to_seq: SeqToSeq,
    epochs: int,
    optim: torch.optim.Optimizer,
    train_loader: DataLoader,
    eval_loader: Optional[DataLoader] = None,
    callbacks: Optional[Collection[Callback]] = None,
) -> None:
    r"""Fit ``seq_to_seq`` with ``optim`` for ``epochs`` ``train_loader`` iters.

    Args:
        seq_to_seq: A :py:class:`.SeqToSeq`.

        epochs: Maximum number of epochs to train ``seq_to_seq`` for. Note that
            the actual number of epochs may be less if
            :py:meth:`.CallbackHandler.on_epoch_end` returns :py:data:`True`.

        optim: A :py:class:`torch.optim.Optimizer` initialized with
            ``seq_to_seq.model`` params.

        train_loader: A :py:class:`torch.utils.data.DataLoader` for the
            training data.

        eval_loader: An optional :py:class:`torch.utils.data.DataLoader` for
            the validation data.

        callbacks: A collection of :py:class:`.Callback`\s.
    """
    # sphinx-doc-start-after
    cb_handler = CallbackHandler(callbacks)
    cb_handler.on_train_begin(epochs)

    for epoch in range(epochs):
        cb_handler.on_epoch_begin()

        stages = [Stage.TRAIN]  # always train for an epoch
        if eval_loader is not None:
            stages.append(Stage.EVAL)  # eval after every epoch
            if epoch == 0:
                stages.insert(0, Stage.EVAL)  # eval before training starts

        for stage in stages:
            is_training = stage == Stage.TRAIN
            seq_to_seq.train(mode=is_training)
            cb_handler.train(mode=is_training)

            with ExitStack() as stack:
                if not is_training:
                    stack.enter_context(torch.no_grad())

                loader = train_loader if is_training else eval_loader
                for x, y in loader:
                    x, y = cb_handler.on_batch_begin(x, y)
                    out = seq_to_seq.model(**x)

                    out = cb_handler.on_loss_begin(out)
                    loss = seq_to_seq.loss(**out, **y)
                    loss, skip_bwd = cb_handler.on_backward_begin(loss)

                    if is_training:
                        if not skip_bwd:
                            loss.backward()

                        if not cb_handler.on_backward_end():
                            optim.step()

                        if not cb_handler.on_step_end():
                            optim.zero_grad()

                    if cb_handler.on_batch_end():
                        break

        if cb_handler.on_epoch_end():
            break

    cb_handler.on_train_end()
    # sphinx-doc-end-before
