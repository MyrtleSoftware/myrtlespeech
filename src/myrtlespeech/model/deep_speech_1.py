from typing import Optional
from typing import Tuple

import torch
from myrtlespeech.model.rnn import RNN
from myrtlespeech.model.rnn import RNNState
from myrtlespeech.model.rnn import RNNType


class DeepSpeech1(torch.nn.Module):
    """A `Deep Speech 1 <https://arxiv.org/abs/1412.5567>`_ -like model.

    Args:
        in_features: Number of input features per step per batch.

        n_hidden: Internal hidden unit size.

        out_features: Number of output features per step per batch.

        drop_prob: Dropout drop probability.

        relu_clip: ReLU clamp value: `min(max(0, x), relu_clip)`.

        forget_gate_bias: Total initialized value of the bias used in the
            forget gate. Set to :py:data:`None` to use PyTorch's default
            initialisation.  (For more information, see `An Empirical
            Exploration of Recurrent Network Architectures
            <http://proceedings.mlr.press/v37/jozefowicz15.pdf>`_)

    Example:
        >>> ds1 = DeepSpeech1(
        ...     in_features=5,
        ...     n_hidden=10,
        ...     out_features=26,
        ...     drop_prob=0.25,
        ...     relu_clip=10.0,
        ...     forget_gate_bias=1.0
        ... )
        >>> ds1
        DeepSpeech1(
          (fc1): Sequential(
            (0): Linear(in_features=5, out_features=10, bias=True)
            (1): Hardtanh(min_val=0.0, max_val=10.0, inplace=True)
            (2): Dropout(p=0.25, inplace=False)
          )
          (fc2): Sequential(
            (0): Linear(in_features=10, out_features=10, bias=True)
            (1): Hardtanh(min_val=0.0, max_val=10.0, inplace=True)
            (2): Dropout(p=0.25, inplace=False)
          )
          (fc3): Sequential(
            (0): Linear(in_features=10, out_features=20, bias=True)
            (1): Hardtanh(min_val=0.0, max_val=10.0, inplace=True)
            (2): Dropout(p=0.25, inplace=False)
          )
          (bi_lstm): RNN(
            (rnn): LSTM(20, 10, batch_first=True, bidirectional=True)
          )
          (fc4): Sequential(
            (0): Linear(in_features=20, out_features=10, bias=True)
            (1): Hardtanh(min_val=0.0, max_val=10.0, inplace=True)
            (2): Dropout(p=0.25, inplace=False)
          )
          (out): Linear(in_features=10, out_features=26, bias=True)
        )
    """

    def __init__(
        self,
        in_features: int,
        n_hidden: int,
        out_features: int,
        drop_prob: float,
        relu_clip: float = 20.0,
        forget_gate_bias: float = 1.0,
    ):
        super().__init__()

        self.use_cuda = torch.cuda.is_available()

        self._relu_clip = float(relu_clip)
        self._drop_prob = drop_prob

        self.fc1 = self._fully_connected(in_features, n_hidden)
        self.fc2 = self._fully_connected(n_hidden, n_hidden)
        self.fc3 = self._fully_connected(n_hidden, 2 * n_hidden)
        self.bi_lstm = RNN(
            rnn_type=RNNType.LSTM,
            input_size=2 * n_hidden,
            hidden_size=n_hidden,
            num_layers=1,
            bias=True,
            bidirectional=True,
            forget_gate_bias=forget_gate_bias,
            batch_first=True,
        )
        self.fc4 = self._fully_connected(2 * n_hidden, n_hidden)
        self.out = self._fully_connected(
            n_hidden, out_features, relu=False, dropout=False
        )
        if self.use_cuda:
            self.cuda()

    def _fully_connected(
        self, in_f: int, out_f: int, relu: bool = True, dropout: bool = True
    ) -> torch.nn.Module:
        layers = [torch.nn.Linear(in_f, out_f)]
        if relu:
            layers.append(
                torch.nn.Hardtanh(0.0, self._relu_clip, inplace=True)
            )
        if dropout:
            layers.append(torch.nn.Dropout(p=self._drop_prob))
        if len(layers) == 1:
            return layers[0]
        return torch.nn.Sequential(*layers)

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        hx: Optional[RNNState] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], RNNState]:
        r"""Returns result of applying the model to ``x[0]`` + lstm state.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args
            x: A tuple where the first element is the network input (a
                :py:class:`torch.Tensor`) with size ``[batch, channels,
                features, seq_len]`` and the second element is
                :py:class:`torch.Tensor` of size ``[batch]`` where each entry
                represents the sequence length of the corresponding *input*
                sequence.

            hx: An Optional RNNState of ``self.bi_lstm`` (see
                :py:class:`RNN` for details.)

        Returns:
            A ``Tuple[Tuple[output, lengths], state]``. ``output`` is the
            result after applying the model to ``x[0]``. It must have size
            ``[seq_len, batch, out_features]``. ``lengths`` is a
            :py:class:`torch.Tensor` with size ``[batch]`` where each entry
            represents the sequence length of the corresponding *output*
            sequence. This will be equal to ``x[1]`` as this layer does not
            currently change sequence length. ``state`` is the hidden state of
            ``self.bi_lstm`` (see :py:class:`RNN` for details.).
        """
        h, seq_lens = x
        if self.use_cuda:
            h = h.cuda()
            seq_lens = seq_lens.cuda()

        batch, channels, features, seq_len = h.size()
        h = h.view(batch, channels * features, seq_len).permute(0, 2, 1)

        h = self.fc1(h)
        h = self.fc2(h)
        h = self.fc3(h)

        (h, seq_lens), hid = self.bi_lstm(x=(h, seq_lens), hx=hx)

        h = self.fc4(h)
        out = self.out(h)
        out = out.transpose(0, 1)

        return (out, seq_lens), hid
