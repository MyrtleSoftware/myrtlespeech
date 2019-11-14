from contextlib import ExitStack
from typing import Collection
from typing import Optional

import torch
from myrtlespeech.model.seq_to_seq import SeqToSeq
from myrtlespeech.run.callbacks.callback import Callback
from myrtlespeech.run.callbacks.callback import CallbackHandler
from myrtlespeech.run.stage import Stage
from torch.utils.data import DataLoader


def fit(
    seq_to_seq: SeqToSeq,
    epochs: int,
    train_loader: DataLoader,
    eval_loader: Optional[DataLoader] = None,
    callbacks: Optional[Collection[Callback]] = None,
) -> None:
    r"""Fit ``seq_to_seq`` for ``epochs`` ``train_loader`` iters.

    Args:
        seq_to_seq: A :py:class:`.SeqToSeq` model.

        epochs: Maximum number of epochs to train ``seq_to_seq`` for. Note that
            the actual number of epochs may be less if
            :py:meth:`.CallbackHandler.on_epoch_end` returns :py:data:`True`.

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
        stages = [Stage.TRAIN]  # always train for an epoch
        if eval_loader is not None:
            stages.append(Stage.EVAL)  # eval after every epoch
            if epoch == 0:
                stages.insert(0, Stage.EVAL)  # eval before training starts

        for stage in stages:
            is_training = stage == Stage.TRAIN
            loader = train_loader if is_training else eval_loader
            if run_stage(seq_to_seq, cb_handler, loader, is_training):
                break

    cb_handler.on_train_end()
    # sphinx-doc-end-before


def run_stage(seq_to_seq, cb_handler, loader, is_training):
    seq_to_seq.train(mode=is_training)
    cb_handler.train(mode=is_training)

    cb_handler.on_epoch_begin()

    with ExitStack() as stack:
        if not is_training:
            stack.enter_context(torch.no_grad())

        for x, y in loader:

            x, y = cb_handler.on_batch_begin(x, y)
            out = seq_to_seq.model(x)

            # loss
            loss_out, loss_y = cb_handler.on_loss_begin(out, y)
            loss = seq_to_seq.loss(loss_out, loss_y)
            loss, skip_bwd = cb_handler.on_backward_begin(loss)

            # optim
            if is_training:
                if not skip_bwd:
                    loss.backward()

                if is_training and seq_to_seq.lr_scheduler is not None:
                    seq_to_seq.lr_scheduler.step()

            if cb_handler.on_epoch_end():
                break

            del x, y, out, loss, loss_out, loss_y

    return cb_handler.on_epoch_end()
