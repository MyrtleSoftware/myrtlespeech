from typing import Optional
from typing import Tuple

import torch


class DeepSpeech2(torch.nn.Module):
    r"""`Deep Speech 2 <http://proceedings.mlr.press/v48/amodei16.pdf>`_ model.

    Args:
        cnn: A :py:class:`torch.nn.Module` containing the convolution part of
            the Deep Speech 2 model.

            Must accept as input a tuple where the first element is the network
            input (a :py:class:`torch.Tensor`) with size ``[batch, channels,
            features, max_input_seq_len]`` and the second element is a
            :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represents the sequence length of the corresponding *input*
            sequence to the cnn.

            It must return a tuple where the first element is the result after
            applying the module to the network input. It must have size
            ``[batch, out_channels, out_features, max_cnn_seq_len]``. The
            second element of the tuple return value is a
            :py:class:`torch.Tensor` with size ``[batch]`` where each entry
            represents the sequence length of the corresponding *output*
            sequence. These may be different than the input sequence lengths
            due to striding and pooling.

        rnn: A :py:class:`torch.nn.Module` containing the recurrent part of
            the Deep Speech 2 model.

            Must accept as input a tuple where the first element is the network
            input (a :py:class:`torch.Tensor`) with size ``[max_cnn_seq_len,
            batch, out_channels*out_features]`` and the second element is a
            :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represents the sequence length of the corresponding *input*
            sequence to the rnn.

            It must return a tuple where the first element is the result after
            applying the module to the cnn output. It must have size
            ``[max_rnn_seq_len, batch, rnn_features]``. The second element of
            the tuple return value is a :py:class:`torch.Tensor` with size
            ``[batch]`` where each entry represents the sequence length of the
            corresponding *output* sequence. These may be different than the
            input sequence lengths due to downsampling.

        lookahead: An optional :py:class:`torch.nn.Module` containing the
            lookahead part of the Deep Speech 2 model. If :py:data:`None` then
            no lookahead is applied (e.g. not required when using bidirectional
            rnns).

            If not :py:data:`None` it must accept as input a tuple where the
            first element is the network input (a :py:class:`torch.Tensor`)
            with size ``[batch, rnn_features, max_rnn_seq_len]`` and the second
            element is a :py:class:`torch.Tensor` of size ``[batch]`` where
            each entry represents the sequence length of the corresponding
            *input* sequence to the lookahead layer.

            It must return a tuple where the first element is the result after
            applying the module to the rnn output. It must have size
            ``[batch, la_features, max_la_seq_len]``. The second element of the
            tuple return value is a :py:class:`torch.Tensor` with size
            ``[batch]`` where each entry represents the sequence length of the
            corresponding *output* sequence. These will usually be equal to the
            input sequence lengths.

        fully_connected: A :py:class:`torch.nn.Module` containing the fully
            connected part of the Deep Speech 2 model.

            Must accept as input a tuple where the first element is the network
            input (a :py:class:`torch.Tensor`) with size ``[batch,
            max_fc_in_seq_len, max_fc_in_features]`` and the second element is
            a :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represents the sequence length of the corresponding *input*
            sequence to the fully connected layer(s). ``max_fc_in_seq_len`` and
            ``max_fc_in_features`` will either be ``max_rnn_seq_len`` and
            ``rnn_features`` or ``max_la_seq_len`` and ``la_features``
            depending on whether lookahead is :py:data:`None`.

            It must return a tuple where the first element is the result after
            applying the module to the previous layers output. It must have
            size ``[batch, max_out_seq_len, out_features]``. The second element
            of the tuple return value is a :py:class:`torch.Tensor` with size
            ``[batch]`` where each entry represents the sequence length of the
            corresponding *output* sequence.
    """

    def __init__(
        self,
        cnn: torch.nn.Module,
        rnn: torch.nn.Module,
        lookahead: Optional[torch.nn.Conv1d],
        fully_connected: torch.nn.Module,
    ):
        super().__init__()

        self.cnn = cnn
        self.rnn = rnn
        self.lookahead = lookahead
        self.fully_connected = fully_connected

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            if self.cnn is not None:
                self.cnn = self.cnn.cuda()
            self.rnn = self.rnn.cuda()
            if self.lookahead is not None:
                self.lookahead = self.lookahead.cuda()
            self.fully_connected = self.fully_connected.cuda()

    def _conv_to_rnn_size(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, chnls, feature, seq_len)->(seq_len, batch, chnls*feature)"""
        batch, channels, features, seq_len = x.size()
        return x.view(batch, channels * features, seq_len).permute(2, 0, 1)

    def _rnn_to_lookahead_size(self, x: torch.Tensor) -> torch.Tensor:
        """(seq_len, batch, features) -> (batch, features, seq_len)"""
        return x.permute(1, 2, 0)

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the result of applying the Deep Speech 2 model to the input.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        See :py:class:`.DeepSpeech2` for detailed information about the input
        and output of each module.

        Args:
            x: Input for the cnn module. See initialisation docstring.

        Returns:
            Output from the fully connected layer but with the ``batch`` and
            ``max_out_seq_len`` dimensions transposed. See initialisation
            docstring.
        """
        h = x

        if self.use_cuda:
            h = (h[0].cuda(), h[1].cuda())

        if self.cnn is not None:
            h = self.cnn(h)
        h = (self._conv_to_rnn_size(h[0]), h[1])

        h, _ = self.rnn(x=h)

        if self.lookahead is not None:
            h = (self._rnn_to_lookahead_size(h[0]), h[1])
            h = self.lookahead(h)
            h = (h[0].transpose(1, 2), h[1])
        else:
            h = (h[0].transpose(0, 1), h[1])

        h = self.fully_connected(h)

        h = (h[0].transpose(0, 1), h[1])

        return h
