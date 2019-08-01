from typing import Optional
from typing import Tuple
from typing import Union

import torch


class Decoder(torch.nn.Module):
    r"""A base class for the decoder part of an :py:class:`.EncoderDecoder`.

    Subclasses must implement :py:meth:`Decoder.forward`.
    """

    def forward(
        self, x: torch.Tensor, seq_lens: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Returns the result of applying the ``Decoder`` to ``x``.

        Args:
            x: This is a :py:class:`torch.Tensor` with size ``[max_out_seq_len,
                batch, out_features]``.

            seq_lens: An optional :py:class:`torch.Tensor` of size ``[batch]``
                where each entry represents the sequence length of the
                corresponding *input* sequence in ``x``.

        Returns:
            A Tuple is returned when ``seq_lens`` is not None.

            The single return value or first element of the Tuple return value
            is the result after applying the :py:class:`Decoder` to ``x``. It
            must have size ``[max_out_seq_len, batch, out_features]``.

            The second element of the Tuple return value is a
            :py:class:`torch.Tensor` with size ``[batch]`` where each entry
            represents the sequence length of the corresponding *output*
            sequence. Each of these will be less than or equal to
            ``max_out_seq_len``.
        """
        raise NotImplementedError()
