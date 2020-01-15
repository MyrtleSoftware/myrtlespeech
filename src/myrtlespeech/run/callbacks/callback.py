from typing import Any
from typing import Collection
from typing import Dict
from typing import Optional
from typing import Tuple

import torch


class Callback:
    r"""Base class for callbacks where all methods do nothing.

    Each method is called by the :py:class:`CallbackHandler` at specific times
    during the training loop -- see :py:func:`.run.train.fit` -- with the
    :py:data:`CallbackHandler.state_dict` passed as ``**kwargs``.

    Each method may optionally return a dictionary. If a key in the returned
    dictionary is present in the :py:data:`CallbackHandler.state_dict` then the
    ``state_dict`` will be updated with the corresponding value. An error is
    raised if not.

    Args:
        training: See Attributes.

    Attributes:
        training: :py:data:`True` when the :py:class:`CallbackHandler` is in
            training mode.
    """

    def __init__(self, training: bool = True):
        self.training = training

    def on_train_begin(self, **kwargs) -> Optional[Dict]:
        ...

    def on_epoch_begin(self, **kwargs) -> Optional[Dict]:
        ...

    def on_batch_begin(self, **kwargs) -> Optional[Dict]:
        ...

    def on_loss_begin(self, **kwargs) -> Optional[Dict]:
        ...

    def on_backward_begin(self, **kwargs) -> Optional[Dict]:
        ...

    def on_backward_end(self, **kwargs) -> Optional[Dict]:
        ...

    def on_step_end(self, **kwargs) -> Optional[Dict]:
        ...

    def on_batch_end(self, **kwargs) -> Optional[Dict]:
        ...

    def on_epoch_end(self, **kwargs) -> Optional[Dict]:
        ...

    def on_train_end(self, **kwargs) -> Optional[Dict]:
        ...

    def train(self, mode=True):
        """Sets the callback in training mode.

        Returns:
            self
        """
        self.training = mode
        return self


class ModelCallback(Callback):
    """Base class for callbacks that need access to the model.

    Args:
        model: See Attributes.
        training: See Attributes.

    Attributes:
        model: A :py:class:`torch.nn.Module`.
        training: See :py:class:`.Callback`.
    """

    def __init__(self, model: torch.nn.Module, training: bool = True):
        super().__init__(training=training)
        self.model = model


class CallbackHandler:
    r"""Manages all registered :py:class:`Callback`\s.

    Args:
        callbacks: A collection of :py:class:`Callback`\s.

        training: See Attributes.

        epoch: training epoch at :py:class:`CallbackHandler` initialization.
            Useful if resuming a training run.

        total_train_batches: number of batches seen during training when a
            training run is resumed.

    Attributes:
        state_dict: A dictionary containing the state of the
            :py:class:`CallbackHandler`.

        training: :py:data:`True` when the :py:class:`CallbackHandler` is in
            training mode.
    """

    def __init__(
        self,
        callbacks: Optional[Collection[Callback]] = None,
        training: bool = True,
        epoch: Optional[int] = None,
        total_train_batches: Optional[int] = None,
    ):
        self.callbacks = callbacks if callbacks is not None else []
        self.state_dict: Dict = {}
        self.training = training
        self.epoch = epoch or 0
        self.total_train_batches = total_train_batches or 0

    def __call__(self, stage_name: str) -> None:
        r"""Runs the ``stage_name`` method of all :py:class:`Callback`\s.

        The ``stage_name`` method of each callback is called with the
        :py:data:`CallbackHandler.state_dict` given as ``**kwargs``.  If a
        :py:class:`Callback` returns a dictionary then it is used to update the
        :py:data:`CallbackHandler.state_dict`. All keys in the returned
        dictionary *must* be present in the
        :py:data:`CallbackHandler.state_dict`.

        Args:
            stage_name: The name of a callback stage (i.e. any
                :py:class:`Callback` method name).

        Raises:
            :py:class:`ValueError`: If any key in any returned dictionary is
                not already present in the
                :py:data:`CallbackHandler.state_dict`.

        Example:
            >>> callback_1 = Callback()
            >>> callback_1.on_begin_epoch = lambda: print('1')
            >>> callback_2 = Callback()
            >>> callback_2.on_begin_epoch = lambda: print('2')
            >>> handler = CallbackHandler(callbacks=[callback_1, callback_2])
            >>> handler('on_begin_epoch')
            1
            2
        """
        for callback in self.callbacks:
            new = getattr(callback, stage_name)(**self.state_dict)
            if new is None:
                continue
            for k, v in new.items():
                if k not in self.state_dict:
                    raise Exception(
                        f"{k} is not a valid key in CallbackHandler state."
                    )
                self.state_dict[k] = v

    def on_train_begin(self, epochs: int) -> None:
        """Initializes :py:data:`CallbackHandler.state_dict` and runs callbacks.

        The initialized :py:data:`CallbackHandler.state_dict` will contain the
        following keys:

            epoch:
                An :py:class:`int` denoting the total number of training epochs
                done so far. Initialized to zero.

            epochs:
                An :py:class:`int` denoting the total number of training
                epochs. Initialized to the ``epochs`` argument.

            total_train_batches:
                TODO
                An :py:class:`int` denoting the total number of calls to
                :py:meth:`CallbackHandler.on_batch_end` since the last call to
                :py:meth:`CallbackHandler.on_train_begin`.

            epoch_batches:
                TODO
                An :py:class:`int` denoting the total number of calls to
                :py:meth:`CallbackHandler.on_batch_end` since the last call to
                :py:meth:`CallbackHandler.on_epoch_begin`.

            reports:
                TODO

        Example:
            >>> # noqa: E501
            >>> handler = CallbackHandler(callbacks=[])
            >>> handler.state_dict
            {}
            >>> handler.on_train_begin(epochs=100)
            >>> handler.state_dict
            {'epoch': 0, 'epochs': 100, 'total_train_batches': 0, 'epoch_batches': 0, 'reports': {}}
        """
        self.state_dict.update(
            dict(
                epoch=self.epoch,
                epochs=epochs,
                total_train_batches=self.total_train_batches,
                epoch_batches=0,
                reports={},
            )
        )
        # delete training state as this will quickly be out of date:
        del self.epoch
        del self.total_train_batches
        self("on_train_begin")

    def on_epoch_begin(self) -> None:
        """Sets ``epoch_batches=0`` in ``state_dict`` and runs callbacks.

        Example:
            >>> callback = Callback()
            >>> callback.on_epoch_begin = lambda **kwargs: print("called")
            >>> handler = CallbackHandler(callbacks=[callback])
            >>> handler.state_dict
            {}
            >>> handler.on_epoch_begin()
            called
            >>> handler.state_dict
            {'epoch_batches': 0}
        """
        self.state_dict["epoch_batches"] = 0
        self("on_epoch_begin")

    def on_batch_begin(self, x: Any, y: Any) -> Tuple[Dict, Dict]:
        """Updates ``state_dict``, runs callbacks, and returns inputs and target.

        The following keys are first set in
        :py:data:`CallbackHandler.state_dict`:

            last_input:
                Set to ``x``.

            last_target:
                Set to ``y``.

        All :py:meth:`Callback.on_batch_begin` methods are then ran. These may
        modify the ``last_input`` and ``last_target``
        :py:data:`CallbackHandler.state_dict` values.

        The possibly modified ``last_input`` and ``last_target`` values are
        then returned.

        Returns:
            Possibly modified ``last_input`` and ``last_target`` values.

        Example:
            >>> # noqa: E501
            >>> callback = Callback()
            >>> callback.on_batch_begin = lambda **kwargs: {'last_input': {'foo': 1.0}}
            >>> handler = CallbackHandler(callbacks=[callback])
            >>> handler.on_batch_begin(x={'x': 0.0}, y={'y': 0.0})
            ({'foo': 1.0}, {'y': 0.0})
        """
        self.state_dict["last_input"] = x
        self.state_dict["last_target"] = y
        self("on_batch_begin")
        return self.state_dict["last_input"], self.state_dict["last_target"]

    def on_loss_begin(self, out: Any, y: Any) -> Tuple[Dict, Dict]:
        """Updates ``state_dict``, runs callbacks, and returns output.

        The following key is first set in
        :py:data:`CallbackHandler.state_dict`:

            last_output:
                Set to ``out``.

            last_target:
                Set to ``y``.

        All :py:meth:`Callback.on_loss_begin` methods are then ran. These may
        modify the ``last_output`` and ``last_target``
        :py:data:`CallbackHandler.state_dict` value.

        The possibly modified ``last_output`` and ``last_target`` is then
        returned.

        Returns:
            Possibly modified ``last_output`` and ``last_target`` value.

        Example:
            TODO
        """
        self.state_dict["last_output"] = out
        self.state_dict["last_target"] = y

        self.state_dict["loss"] = {
            "last_output": self.state_dict["last_output"],
            "last_target": self.state_dict["last_target"],
        }
        self("on_loss_begin")

        return (
            self.state_dict["loss"]["last_output"],
            self.state_dict["loss"]["last_target"],
        )

    def on_backward_begin(
        self, loss: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        """Updates ``state_dict``, runs callbacks, and returns loss and bool.

        The following keys are first set in
        :py:data:`CallbackHandler.state_dict`:

            skip_bwd:
                Set to ``False``.

            last_loss:
                Set to ``loss``.

        All :py:meth:`Callback.on_backward_begin` methods are then ran. These
        may modify the ``skip_bwd`` and ``last_loss``
        :py:data:`CallbackHandler.state_dict` values.

        The possibly modified ``last_loss`` and ``skip_bwd`` values are then
        returned.

        Returns:
            Possibly modified ``last_loss`` and ``skip_bwd`` values.

        Example:
            >>> # noqa: E501
            >>> callback = Callback()
            >>> callback.on_backward_begin = lambda **kwargs: {"skip_bwd": True, "last_loss": torch.tensor([0.0])}
            >>> handler = CallbackHandler(callbacks=[callback])
            >>> handler.on_backward_begin(loss=torch.tensor([1.0]))
            (tensor([0.]), True)
        """
        self.state_dict["skip_bwd"] = False
        self.state_dict["last_loss"] = loss
        self("on_backward_begin")
        return self.state_dict["last_loss"], self.state_dict["skip_bwd"]

    def on_backward_end(self) -> bool:
        """Updates ``state_dict``, runs callbacks, and returns a bool.

        The following key is first set in
        :py:data:`CallbackHandler.state_dict`:

            skip_step:
                Set to ``False``.

        All :py:meth:`Callback.on_backward_end` methods are then ran. These may
        modify the ``skip_step`` :py:data:`CallbackHandler.state_dict` value.

        The possibly modified ``skip_step`` value is then returned.

        Returns:
            Possibly modified ``skip_step`` value.

        Example:
            >>> callback = Callback()
            >>> callback.on_backward_end = lambda **kwargs: {"skip_step": True}
            >>> handler = CallbackHandler(callbacks=[callback])
            >>> handler.on_backward_end()
            True
        """
        self.state_dict["skip_step"] = False
        self("on_backward_end")
        return self.state_dict["skip_step"]

    def on_step_end(self) -> bool:
        """Updates ``state_dict``, runs callbacks, and returns a bool.

        The following key is first set in
        :py:data:`CallbackHandler.state_dict`:

            skip_zero:
                Set to ``False``.

        All :py:meth:`Callback.on_step_end` methods are then ran. These may
        modify the ``skip_zero`` :py:data:`CallbackHandler.state_dict` value.

        The possibly modified ``skip_zero`` value is then returned.

        Returns:
            Possibly modified ``skip_zero`` value.

        Example:
            >>> callback = Callback()
            >>> callback.on_step_end = lambda **kwargs: {"skip_zero": True}
            >>> handler = CallbackHandler(callbacks=[callback])
            >>> handler.on_step_end()
            True
        """
        self.state_dict["skip_zero"] = False
        self("on_step_end")
        return self.state_dict["skip_zero"]

    def on_batch_end(self) -> bool:
        """Updates ``state_dict``, runs callbacks, and returns a bool.

        The following key is first set in
        :py:data:`CallbackHandler.state_dict`:

            stop_epoch:
                Set to ``False``.

        All :py:meth:`Callback.on_batch_end` methods are then ran. These may
        modify the ``stop_epoch`` :py:data:`CallbackHandler.state_dict` value.

        If :py:data:`CallbackHandler.training` then the following keys are then
        modified in :py:data:`CallbackHandler.state_dict`:

            total_train_batches:
                Incremented by 1 if ``self.training``.

            epoch_batches:
                Incremented by 1.

        The possibly modified ``stop_epoch`` value is then returned.

        Returns:
            Possibly modified ``stop_epoch`` value.

        Example:
            >>> # noqa: E501
            >>> callback = Callback()
            >>> callback.on_batch_end = lambda **kwargs: {"stop_epoch": True}
            >>> handler = CallbackHandler(callbacks=[callback])
            >>> handler.on_train_begin(1)   # initialise state_dict
            >>> handler.state_dict
            {'epoch': 0, 'epochs': 1, 'total_train_batches': 0, 'epoch_batches': 0, 'reports': {}}
            >>> handler.on_batch_end()
            True
            >>> handler.state_dict
            {'epoch': 0, 'epochs': 1, 'total_train_batches': 1, 'epoch_batches': 1, 'reports': {}, 'stop_epoch': True}
        """
        self.state_dict["stop_epoch"] = False
        self("on_batch_end")
        self.state_dict["epoch_batches"] += 1
        if self.training:
            self.state_dict["total_train_batches"] += 1
        return self.state_dict["stop_epoch"]

    def on_epoch_end(self) -> bool:
        r"""Updates ``state_dict``, runs callbacks, and returns a bool.

        The following key is first set in
        :py:data:`CallbackHandler.state_dict`\:

            stop_training:
                Set to ``False``.

        All :py:meth:`Callback.on_epoch_end` methods are then ran. These may
        modify the ``stop_training`` :py:data:`CallbackHandler.state_dict`
        value.

        The following key is then modified in
        :py:data:`CallbackHandler.state_dict`\:

            epoch:
                Incremented by 1 if ``self.training``.

        The possibly modified ``stop_training`` value is then returned.

        Returns:
            Possibly modified ``stop_training`` value.

        Example:
            >>> # noqa: E501
            >>> callback = Callback()
            >>> callback.on_epoch_end = lambda **kwargs: {"stop_training": True}
            >>> handler = CallbackHandler(callbacks=[callback])
            >>> handler.on_train_begin(1)   # initialise state_dict
            >>> handler.state_dict
            {'epoch': 0, 'epochs': 1, 'total_train_batches': 0, 'epoch_batches': 0, 'reports': {}}
            >>> handler.on_epoch_end()
            True
            >>> handler.state_dict["epoch"]
            1
            >>> handler.state_dict["stop_training"]
            True
        """
        self.state_dict["stop_training"] = False
        self("on_epoch_end")
        if self.training:
            self.state_dict["epoch"] += 1
        return self.state_dict["stop_training"]

    def on_train_end(self) -> None:
        """Runs all ``on_train_end`` callbacks."""
        self("on_train_end")

    def train(self, mode=True) -> "CallbackHandler":
        """Sets the handler and all registered callbacks in training mode.

        Returns:
            self
        """
        self.training = mode
        for callback in self.callbacks:
            callback.train(mode=mode)
        return self
