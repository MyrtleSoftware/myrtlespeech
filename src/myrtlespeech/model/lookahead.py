import math

import torch


class Lookahead(torch.nn.Module):
    r"""A lookahead convolution.

    As defined in section 3.5 of `Deep Speech 2: End-to-End Speech Recognition
    in English and Mandarin <http://proceedings.mlr.press/v48/amodei16.pdf>`_.

    Args:
        in_features: Size of each input sample (denoted :math:`d` in the
            paper).

        context: Number of activation timesteps to linearly combine
            (:math:`\tau` in the paper).
    """

    def __init__(self, in_features: int, context: int):
        super().__init__()
        self.in_features = in_features
        self.context = context
        self.weight = torch.nn.Parameter(
            torch.Tensor(self.in_features, 1, self.context)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Returns the result of applying the lookahead layer to ``x``.

        Args:
            x: A :py:class:`torch.Tensor` with size ``(batch, in_features,
                seq_len)``.

        Returns:
            A :py:class:`torch.Tensor` with size ``(batch,  in_features,
            out_seq_len)`` where ``out_seq_len = seq_len - context + 1``.
        """
        return torch.nn.functional.conv1d(
            input=x, weight=self.weight, groups=self.in_features
        )

    def extra_repr(self) -> str:
        """See :py:meth:`torch.nn.Module.extra_repr`."""
        return f"in_features={self.in_features}, context={self.context}"
