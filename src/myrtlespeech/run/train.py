from contextlib import ExitStack
from typing import Collection
from typing import Dict
from typing import List
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
    training_state: Dict = {},
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

        callbacks: A collection of :py:class:`.Callback`\s. If ``hasattr(
            seq_to_seq.model, 'callbacks')``, or ``hasattr(train_loader,
            'callbacks')`` these callbacks are added to the ``callbacks``
            collection.

        training_state: A dictionary containing the training state returned
            by :py:func:`load_state_dict`.
    """
    assert (
        train_loader or eval_loader
    ), "Can't have both train_loader==None and eval_loader==None"
    if train_loader is None:
        assert epochs == 1, "If train_loader is None, epochs must be 1 "
    start_epoch = training_state.get("epoch", 0)
    if start_epoch > epochs:
        raise ValueError(
            f'training_state["epoch"] is greater than ``epochs``'
            f"so no training can occur."
        )
    callbacks = _extend_callbacks(callbacks, seq_to_seq.model, train_loader)
    print(callbacks)
    # sphinx-doc-start-after
    cb_handler = CallbackHandler(callbacks, **training_state)
    cb_handler.on_train_begin(epochs)

    for epoch in range(start_epoch, epochs):
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
                    print("on backwards begin...", loss.item())
                    loss, skip_bwd = cb_handler.on_backward_begin(loss)
                    print("post on backwards begin...", loss.item())
                    # optim
                    if is_training:
                        if not skip_bwd:
                            loss.backward()

                        if seq_to_seq.optim is not None:
                            if not cb_handler.on_backward_end():
                                print("stepping...", loss.item())
                                seq_to_seq.optim.step()
                                seq_to_seq.lr_scheduler.step()

                            if not cb_handler.on_step_end():
                                print("zero grad...", loss.item())
                                seq_to_seq.optim.zero_grad()

                    if cb_handler.on_batch_end():
                        break

            if cb_handler.on_epoch_end():
                break

    cb_handler.on_train_end()
    # sphinx-doc-end-before


def _extend_callbacks(
    callbacks: Optional[Collection[Callback]],
    model: torch.nn.Module,
    train_loader: Optional[DataLoader],
):
    cbs: List = []
    if hasattr(model, "callbacks"):
        cbs = model.callbacks.copy()
    cbs.extend(callbacks or [])
    if train_loader and hasattr(train_loader, "callbacks"):
        cbs.extend(train_loader.callbacks or [])
    return cbs
