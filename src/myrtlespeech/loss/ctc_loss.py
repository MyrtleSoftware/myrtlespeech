from typing import Tuple
from typing import Union

import torch


class CTCLoss(torch.nn.Module):
    """Wrapped :py:class:`torch.nn.CTCLoss` that applies LogSoftmax internally.

    Args:
        blank: Index of the blank label.

        reduction: Specifies the reduction to apply to the output:

            none:
                No reduction will be applied.

            mean:
                The output losses will be divided by the target lengths and
                then the mean over the batch is taken.

            sum:
                Sum all losses in a batch.

        zero_infinity: Whether to zero infinite losses and the associated
            gradients. (Infinite losses mainly occur when the inputs are too
            short to be aligned to the targets)

        dim: A dimension along which :py:class:`torch.nn.LogSoftmax` will be
            computed.

    Attributes:
        log_softmax: A :py:class:`torch.nn.LogSoftmax` instance.

        ctc_loss: A :py:class:`torch.nn.CTCLoss` instance.
    """

    def __init__(
        self,
        blank: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = True,
        dim: int = -1,
    ):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=dim)
        self.ctc_loss = torch.nn.CTCLoss(
            blank=blank, reduction=reduction, zero_infinity=zero_infinity
        )
        self.use_cuda = torch.cuda.is_available()

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: Union[Tuple, torch.Tensor],
        target_lengths: Union[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """Computes CTC loss.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Much of the following documentation is from
        :py:class:`torch.nn.CTCLoss`.

        Args:
            inputs: Unnormalized network :py:class:`torch.Tensor` outputs of
                size ``(max_seq_len, batch, features)``.

            targets: :py:class:`torch.Tensor` where each element in the target
                sequence is a class index. Target indices cannot be the blank
                index.

                Must have size ``(batch, max_seq_len)`` or
                ``sum(target_lengths)``. In the former form each target
                sequence is padded to the length of the longest sequence and
                stacked.

            input_lengths: Length of the inputs (each must be ``<= batch``).
                Lengths are specified for each sequence to achieve masking
                under the assumption that sequences are padded to equal
                lengths.

            target_lengths: Lengths of the targets. Lengths are specified for
                each sequence to achieve masking under the assumption that
                sequences are padded to equal lengths.
        """
        if self.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
            if torch.is_tensor(input_lengths):
                input_lengths = input_lengths.cuda()  # type: ignore
            if torch.is_tensor(target_lengths):
                target_lengths = target_lengths.cuda()  # type: ignore

        log_probs = self.log_softmax(inputs)
        return self.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )
