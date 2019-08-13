from typing import Optional
from typing import Tuple
from typing import Union

import torch
from myrtlespeech.model.encoder_decoder.decoder.decoder import Decoder
from myrtlespeech.model.encoder_decoder.encoder.encoder import Encoder


class EncoderDecoder(torch.nn.Module):
    """An encoder-decoder model.

    All ``encoder`` and ``decoder`` parameters and buffers are moved to the GPU
    with :py:meth:`torch.nn.Module.cuda` if :py:func:`torch.cuda.is_available`.

    Args:
        encoder: An :py:class:`.Encoder`.

        decoder: A :py:class:`.Decoder`.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the result of applying the ``EncoderDecoder`` to ``x[0]``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            x: A tuple where the first element is the network input (a
                :py:class:`torch.Tensor`) with size ``[batch, channels,
                features, max_in_seq_len]`` and the second element is
                :py:class:`torch.Tensor` of size ``[batch]`` where each entry
                represents the sequence length of the corresponding *input*
                sequence.

        Returns:
            The first element of the Tuple return value is the result after
            applying both the :py:class:`.Encoder` and :py:class:`.Decoder` to
            ``x[0]``. It must have size ``[max_out_seq_len, batch,
            out_features]``.

            The second element of the Tuple return value is a
            :py:class:`torch.Tensor` with size ``[batch]`` where each entry
            represents the sequence length of the corresponding *output*
            sequence. Each of these will be less than or equal to
            ``max_out_seq_len``.
        """
        if self.use_cuda:
            x = (x[0].cuda(), x[1].cuda())
        h = self.encoder(x)
        return self.decoder(h)
