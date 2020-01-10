from typing import Tuple

import torch


class StackTime:
    r"""Downsamples sequences in time by stacking in the hidden dimension.

    For example, this Callable may be used before the input to an RNN in order
    to reduce the computational cost.

    Args:
        time_reduction_factor: An int > 0 which gives the factor by which the
            input feature size increases. It is also approximately equal to the
            factor by which the time-sequence length is decreased. For example,
            if `time_reduction_factor` = 2, the input feature size is doubled
            and the sequence length will be approximately halved
            ('approximately' as padding will be required if ``input_seq_len %
            time_reduction_factor != 0``.

        padding_value: Optional integer, default = 0. The padding value that is
            added if ``input_seq_len % time_reduction_factor != 0``.
    """

    def __init__(self, time_reduction_factor, padding_value=0):

        assert isinstance(time_reduction_factor, int)
        assert time_reduction_factor > 0
        self.time_reduction_factor = time_reduction_factor
        self.padding_value = padding_value

    def __call__(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Downsamples sequences in time by stacking in the hidden dimension.

        Args:
            x: A tuple where the first element is a :py:`torch.Tensor` with
                size ``[max_seq_len, batch, in_features]`` and the second
                element is a :py:class:`torch.Tensor` of size ``[batch]`` where
                each entry represents the sequence length of the corresponding
                *input* sequence.

        Returns:
            A tuple where the first element is the result after
                applying the Callable to the output. It will have size
                ``[max_downsampled_seq_len, batch, in_features *
                time_reduction_factor]`` where ``max_downsampled_seq_len`` =
                ``ceil(input_seq_len / time_reduction_factor)``. The second
                element of the tuple is a :py:class:`torch.Tensor` with size
                ``[batch]`` where each entry represents the sequence length of
                the corresponding *output* sequence.
        """

        inp, lens = x

        T, B, P = inp.shape

        if T // self.time_reduction_factor == 0:
            raise ValueError(
                f"Cannot have time_reduction_factor={self.time_reduction_factor} \
            with input sequence length T = {T} since \
            T // time_reduction_factor == 0"
            )

        inp = inp.transpose(0, 1)  # (T, B, P) -> (B, T, P)

        # Add padding if required:
        if T % self.time_reduction_factor != 0:
            # pad end of seq with zeros
            pad = self.time_reduction_factor - T % self.time_reduction_factor
            padding = torch.ones(B, pad, P) * self.padding_value
            padding = padding.to(inp.device).type(inp.dtype)

            inp = torch.cat([inp, padding], dim=1)

        B, T, P = inp.shape
        inp = inp.contiguous()

        inp = inp.view(
            (
                B,
                T // self.time_reduction_factor,
                P * self.time_reduction_factor,
            )
        )

        inp = inp.transpose(0, 1).contiguous()

        lens = torch.ceil(lens.float() / self.time_reduction_factor).int()

        return inp, lens
