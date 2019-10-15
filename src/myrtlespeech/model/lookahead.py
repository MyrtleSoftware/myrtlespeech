import math
from typing import Tuple

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
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.weight.cuda()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns the result of applying the lookahead layer to ``x[0]``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args
            x: A tuple where the first element is the network input (a
                :py:class:`torch.Tensor`) with size ``[batch,
                in_features, seq_len]`` and the second element is
                :py:class:`torch.Tensor` of size ``[batch]`` where each entry
                represents the sequence length of the corresponding *input*
                sequence.

        Returns:
            The first element of the Tuple return value is the result after
            applying the lookahead layer to ``x[0]``. It must have size
            ``[batch, in_features, seq_len]``.

            The second element of the Tuple return value is a
            :py:class:`torch.Tensor` with size ``[batch]`` where each entry
            represents the sequence length of the corresponding *output*
            sequence. This will be equal to ``x[1]`` as this layer does not
            currently change sequence length.
        """
        if self.use_cuda:
            x = (x[0].cuda(), x[1].cuda())
        input = torch.nn.functional.pad(x[0], (0, self.context - 1))
        acts = torch.nn.functional.conv1d(
            input=input, weight=self.weight, groups=self.in_features
        )
        return acts, x[1]

    def extra_repr(self) -> str:
        """See :py:meth:`torch.nn.Module.extra_repr`."""
        return f"in_features={self.in_features}, context={self.context}"
