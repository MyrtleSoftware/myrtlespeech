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
        self, x: torch.Tensor, seq_lens: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Returns the result of applying the ``EncoderDecoder`` to ``x``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            x: A :py:class:`torch.Tensor` with size ``[batch, channels,
                features, max_in_seq_len]``.

            seq_lens: An optional :py:class:`torch.Tensor` of size ``[batch]``
                where each entry represents the sequence length of the
                corresponding *input* sequence in ``x``.

        Returns:
            A Tuple is returned when ``seq_lens`` is not None.

            The single return value or first element of the Tuple return value
            is the result after applying both the :py:class:`.Encoder` and
            :py:class:`.Decoder` to ``x``. It must have size
            ``[max_out_seq_len, batch, out_features]``.

            The second element of the Tuple return value is a
            :py:class:`torch.Tensor` with size ``[batch]`` where each entry
            represents the sequence length of the corresponding *output*
            sequence. Each of these will be less than or equal to
            ``max_out_seq_len``.
        """
        if self.use_cuda:
            x = x.cuda()
            if seq_lens is not None:
                seq_lens = seq_lens.cuda()

        if seq_lens is not None:
            h, seq_lens = self.encoder(x, seq_lens)
            return self.decoder(h, seq_lens)

        h = self.encoder(x)
        return self.decoder(h)
