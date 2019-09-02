from typing import Optional
from typing import Tuple

import torch

from myrtlespeech.model.encoder_decoder.encoder.encoder import conv_to_rnn_size


class DeepSpeech2(torch.nn.Module):
    """`Deep Speech 2 <http://proceedings.mlr.press/v48/amodei16.pdf>`_ model.

    Args:
        cnn: A :py:class:`torch.nn.Module` containing the convolution part of
            the Deep Speech 2 model.

            TODO:
                - must accept as input a tuple of (x, seq_lens)
                - x a torch.tensor of shape TODO
                - seq_lens is a torch.tensor of shape TODO

        rnn:

        lookahead:

        fully_connected:





    CNN, RNN, Lookahead, FullyConnected must accept seq_lens argument

        input: -> [batch, channels, features, max_in_seq_len]

    cnn: -> [batch, channels, out_features, max_out_seq_len]

        reshape: -> [seq_len, batch, channels*out_features]

    rnn: -> [seq_len, batch, out_features]

        reshape: -> [batch, out_features, seq_len)

    lookahead: -> [batch, features, seq_len]

        reshape: -> [batch, seq_len, features]

    fully_connected: [batch, seq_len, out_features]

        reshape: -> [seq_len, batch, out_features]
    """
    def __init__(
        self,
        cnn: torch.nn.Module,
        rnn: torch.nn.Module,
        lookahead: Optional[torch.nn.Conv1d],
        fully_connected: torch.nn.Module
    ):
        super().__init__()

        self.cnn = cnn
        self.rnn = rnn
        self.lookahead = lookahead
        self.fully_connected = fully_connected

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            # TODO: self.cuda()?
            if self.cnn is not None:
                self.cnn = self.cnn.cuda()
            self.rnn = self.rnn.cuda()
            if self.lookahead:
                self.lookahead = self.lookahead.cuda()
            self.fully_connected = self.fully_connected.cuda()

    def rnn_to_lookahead_size(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, batch, features = x.size()
        return x.transpose(0, 1).transpose(1, 2)

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        """
        h = x

        if self.use_cuda:
            h = (h[0].cuda(), h[1].cuda())

        if self.cnn is not None:
            h = self.cnn(h)

        h = (conv_to_rnn_size(h[0]), h[1])

        h = self.rnn(h)

        if self.lookahead is not None:
            h = (self.rnn_to_lookahead_size(h[0]), h[1])
            h = self.lookahead(h)
            h = (h[0].transpose(1, 2), h[1])
        else:
            h = (h[0].transpose(0, 1), h[1])

        h = self.fully_connected(h)

        h = (h[0].transpose(0, 1), h[1])

        return h
