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
    epochs: int = 1,
    train_loader: Optional[DataLoader] = None,
    eval_loader: Optional[DataLoader] = None,
    callbacks: Optional[Collection[Callback]] = None,
) -> None:
    r"""Fit ``seq_to_seq`` for ``epochs`` ``train_loader`` iters.

    When ``train_loader=None``, evaluation is performed **once only** with
    ``eval_loader`` and training is not performed.

    Args:
        seq_to_seq: A :py:class:`.SeqToSeq` model.

        epochs: Maximum number of epochs to train ``seq_to_seq`` for. Note that
            the actual number of epochs may be less if
            :py:meth:`.CallbackHandler.on_epoch_end` returns :py:data:`True`.
            Must be :py:data:`1` if ``train_loader=None``.

        train_loader: An Optional :py:class:`torch.utils.data.DataLoader` for
            the training data.

        eval_loader: An optional :py:class:`torch.utils.data.DataLoader` for
            the validation data.

        callbacks: A collection of :py:class:`.Callback`\s.
    """
    assert (
        train_loader or eval_loader
    ), "Can't have both train_loader==None and eval_loader==None"
    if train_loader is None:
        assert epochs == 1, "If train_loader is None, epochs must be 1 "

    # sphinx-doc-start-after
    cb_handler = CallbackHandler(callbacks)
    cb_handler.on_train_begin(epochs)

    for epoch in range(epochs):
        stages = []
        if train_loader is not None:
            if epoch == 0 and eval_loader is not None:
                stages.append(Stage.EVAL)  # eval before training starts
            stages.append(Stage.TRAIN)

        if eval_loader is not None:
            stages.append(Stage.EVAL)  # eval after every epoch

        for stage in stages:
            is_training = stage == Stage.TRAIN
            seq_to_seq.train(mode=is_training)
            cb_handler.train(mode=is_training)

            cb_handler.on_epoch_begin()

            with ExitStack() as stack:
                if not is_training:
                    stack.enter_context(torch.no_grad())

                loader = train_loader if is_training else eval_loader
                for x, y in loader:
                    # model
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

                        if seq_to_seq.optim is not None:
                            if not cb_handler.on_backward_end():
                                seq_to_seq.optim.step()

                            if not cb_handler.on_step_end():
                                seq_to_seq.optim.zero_grad()

                    if cb_handler.on_batch_end():
                        break

                if is_training and seq_to_seq.lr_scheduler is not None:
                    seq_to_seq.lr_scheduler.step()

            if cb_handler.on_epoch_end():
                break

    cb_handler.on_train_end()
    # sphinx-doc-end-before
