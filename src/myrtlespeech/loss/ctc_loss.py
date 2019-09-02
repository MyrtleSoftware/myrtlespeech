import warnings
from typing import Tuple

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
        zero_infinity: bool = False,
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
        inputs: Tuple[torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Computes CTC loss.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Much of the following documentation is from
        :py:class:`torch.nn.CTCLoss`.

        Args:
            inputs: A tuple where the first element is the unnormalized network
                :py:class:`torch.Tensor` outputs of size ``(max_seq_len, batch,
                features)``. The second element is a :py:class:`torch.Tensor`
                that gives the length of the inputs (each must be ``<=
                max_seq_len``). Lengths are specified for each sequence to
                achieve masking under the assumption that sequences are padded
                to equal lengths.

            targets: A tuple where the first element is a
                :py:class:`torch.Tensor` such that each entry in the target
                sequence is a class index. Target indices cannot be the blank
                index. It must have size ``(batch, max_seq_len)`` or
                ``sum(target_lengths)``. In the former form each target
                sequence is padded to the length of the longest sequence and
                stacked.

                The second element is a :py:class:`torch.Tensor` that gives
                the lengths of the targets. Lengths are specified for each
                sequence to achieve masking under the assumption that sequences
                are padded to equal lengths.
        """
        x, x_lens = inputs
        y, y_lens = targets
        if self.use_cuda:
            x = x.cuda()
            x_lens = x_lens.cuda()
            y = y.cuda()
            y_lens = y_lens.cuda()

        target_lens = []
        for batch_idx in range(targets[0].size(0)):
            target_len = targets[1][batch_idx]
            # increase required length by number of blank symbols necessary
            prev = None
            for symbol_idx in targets[0][batch_idx]:
                if prev is not None and symbol_idx == prev:
                    target_len += 1
                prev = symbol_idx
            target_lens.append(target_len)

        target_lens = torch.tensor(
            target_lens,
            dtype=torch.long,
            requires_grad=False
        )

        if torch.any(x_lens < target_lens.cuda()):
            print(x.size(), x_lens, y_lens, target_lens, sep="\n")
            warnings.warn(
                "output sequence lengths less than target sequence lengths"
            )
            print()
            print()

        log_probs = self.log_softmax(x)
        loss = self.ctc_loss(
            log_probs=log_probs,
            targets=y,
            input_lengths=x_lens,
            target_lengths=y_lens,
        )

        if loss == float("inf") or loss == float("-inf"):
            print(f"loss is {loss}")

        return loss
