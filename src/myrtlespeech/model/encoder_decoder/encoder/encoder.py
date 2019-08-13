from typing import Tuple

import torch


class Encoder(torch.nn.Module):
    r"""A base class for the encoder part of an :py:class:`EncoderDecoder`.

    Subclasses must implement :py:meth:`Encoder.forward`.
    """

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns the result of applying the ``Encoder`` to ``x[0]``.

        Args
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
        raise NotImplementedError()


def conv_to_rnn_size(x: torch.Tensor) -> torch.Tensor:
    r"""Returns a 3D :py:class:`torch.Tensor` given a 4D input.

    Args:
        x: :py:class:`torch.Tensor` with size ``[batch, channels, features,
            seq_len]``

    Returns:
        ``x`` but resized to ``[seq_len, batch, channels*features]``
    """
    batch, channels, features, seq_len = x.size()
    return x.view(batch, channels * features, seq_len).permute(2, 0, 1)
