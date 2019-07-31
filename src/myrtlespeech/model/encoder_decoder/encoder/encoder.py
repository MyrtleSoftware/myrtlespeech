from typing import Optional, Tuple, Union

import torch


class Encoder(torch.nn.Module):
    r"""A base class for the encoder part of an :py:class:`EncoderDecoder`.

    Subclasses must implement :py:meth:`Encoder.forward`.
    """

    def forward(
        self, x: torch.Tensor, seq_lens: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Returns the result of applying the ``Encoder`` to ``x``.

        Args:
            x: This is a :py:class:`torch.Tensor` with size ``[batch, channels,
                features, max_in_seq_len]``.

            seq_lens: An optional :py:class:`torch.Tensor` of size ``[batch]``
                where each entry represents the sequence length of the
                corresponding *input* sequence in ``x``.

        Returns:
            A Tuple is returned when ``seq_lens`` is not None.

            The single return value or first element of the Tuple return value
            is the result after applying the :py:class:`Encoder` to ``x``. It
            must have size ``[max_out_seq_len, batch, out_features]``.

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
