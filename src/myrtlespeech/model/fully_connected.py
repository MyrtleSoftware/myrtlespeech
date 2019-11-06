from typing import Optional
from typing import Tuple
from typing import Union

import torch


class FullyConnected(torch.nn.Module):
    r"""A fully connected neural network.

    All parameters and buffers are moved to the GPU with
    :py:meth:`torch.nn.Module.cuda` if :py:func:`torch.cuda.is_available`.

    Args:
        in_features: Size of each input sample.

        out_features: Size of each output sample.

        hidden_activation_fn: The activation function applied after each hidden
            layer, if any.

        batch_norm: If :py:data:`True`, then batch normalization is added.

    Attributes:
        batch_norm: A :py:class:`torch.BatchNorm1d` instance.

        fully_connected (:py:class:`torch.nn.Linear`):
            A :py:class:`torch.nn.Module` that implements the fully connected
            layer specified by the class arguments. It is an instance of
            :py:class:`torch.nn.Linear`.

        activation: the activation function applied after the linear layer, if
        any.

        in_features: See Args.

        out_features: See Args.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_activation_fn: Optional[torch.nn.Module],
        batch_norm: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fully_connected = torch.nn.Linear(in_features, out_features)
        self.batch_norm = torch.nn.BatchNorm1d(out_features) if batch_norm \
            else None
        self.activation = hidden_activation_fn

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.fully_connected = self.fully_connected.cuda()
            self.batch_norm = self.batch_norm.cuda() if batch_norm else None
            self.activation = self.activation.cuda() if hidden_activation_fn \
                else None

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns the result of applying ``fully_connected`` to ``x``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args
            x: A tuple where the first element is the network input (a
                :py:class:`torch.Tensor`) with size ``[batch, max_seq_len,
                in_features]`` and the second element is
                :py:class:`torch.Tensor` of size ``[batch]`` where each entry
                represents the sequence length of the corresponding *input*
                sequence.

        Returns:
            The first element of the Tuple return value is the result after
            applying the module to ``x[0]``. It has size ``[batch, max_seq_len,
            out_features]``.  The second element of the Tuple return value is a
            :py:class:`torch.Tensor` with size ``[batch]`` where each entry
            represents the sequence length of the corresponding *output*
            sequence.
        """
        if self.use_cuda:
            x = (x[0].cuda(), x[1].cuda())

        result = self.fully_connected(x[0])

        if self.batch_norm is not None:
            # Collapses input of dim T*N*H to (T*N)*H and gives it to a batch
            # norm layer.
            # Allows handling of variable sequence lengths and minibatch sizes.
            t, n = result.size(0), result.size(1)
            x_norm = self.batch_norm(result.reshape(t * n, -1))
            result = x_norm.reshape(t, n, -1)

        if self.activation is not None:
            result = self.activation(result)
        return result, x[1]
