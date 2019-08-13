from typing import Tuple

import torch


class Decoder(torch.nn.Module):
    r"""A base class for the decoder part of an :py:class:`.EncoderDecoder`.

    Subclasses must implement :py:meth:`Decoder.forward`.
    """

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns the result of applying the ``Decoder`` to ``x[0]``.

        Args
            x: A tuple where the first element is the network input (a
                :py:class:`torch.Tensor`) with size ``[max_out_seq_len, batch,
                out_features]`` and the second element is
                :py:class:`torch.Tensor` of size ``[batch]`` where each entry
                represents the sequence length of the corresponding *input*
                sequence.

        Returns:
            The first element of the Tuple return value is the result after
            applying both the :py:class:`.Decoder` to ``x[0]``. It must have
            size ``[max_out_seq_len, batch, out_features]``.

            The second element of the Tuple return value is a
            :py:class:`torch.Tensor` with size ``[batch]`` where each entry
            represents the sequence length of the corresponding *output*
            sequence. Each of these will be less than or equal to
            ``max_out_seq_len``.
        """
        raise NotImplementedError()
