from typing import Collection

import torch
from torch.utils.data import DataLoader

from myrtlespeech.run.callback import Callback, CallbackHandler


def fit(
    model: torch.nn.Module,
    epochs: int,
    optim: torch.optim.Optimizer,
    loader: DataLoader,
    callbacks: Collection[Callback],
) -> None:
    r"""Fits the ``model`` for ``epochs`` iters over ``loader`` using ``optim``.

    Args:
        model: A :py:class:`torch.nn.Module`.

        epochs: Maximum number of epochs to train ``model`` for. Note that the
            actual number of epochs may be less if
            :py:meth:`.CallbackHandler.on_epoch_end` returns :py:data:`True`.

        optim: A :py:class:`torch.optim.Optimizer` initialized with ``model``
            params.

        loader: A :py:class:`torch.utils.data.DataLoader`.

        callbacks: A collection of :py:class:`.Callback`\s.
    """
    # sphinx-doc-start-after
    cb_handler = CallbackHandler(callbacks)
    cb_handler.on_train_begin(epochs)

    for epoch in range(epochs):
        model.train()
        cb_handler.on_epoch_begin()

        for x, y in loader:

            x, y = cb_handler.on_batch_begin(x, y)
            out = model(**x)

            out = cb_handler.on_loss_begin(out)
            loss = model.loss(**out, **y)

            loss, skip_bwd = cb_handler.on_backward_begin(loss)
            if not skip_bwd:
                loss.backward()

            if not cb_handler.on_backward_end():
                optim.step()

            if not cb_handler.on_step_end():
                optim.zero_grad()

            if cb_handler.on_batch_end(loss):
                break

        if cb_handler.on_epoch_end():
            break

    cb_handler.on_train_end()
    # sphinx-doc-end-before
