import warnings
from typing import Tuple

import torch
from warprnnt_pytorch import RNNTLoss as WarpRNNTLoss


class RNNTLoss(torch.nn.Module):
    """Wrapped :py:class:`warprnnt_pytorch.RNNTLoss`.

    Args:
        blank: Index of the blank label.

        reduction: (string) Specifies the reduction to apply to the output:

            none:
                No reduction will be applied.

            mean:
                The output losses will be divided by the target lengths and
                then the mean over the batch is taken.

            sum:
                Sum all losses in a batch.

    Attributes:

        rnnt_loss: A :py:class:`warprnnt_pytorch.RNNTLoss` instance.
    """

    def __init__(self, blank: int = 0, reduction: str = "mean"):
        super().__init__()
        self.rnnt_loss = WarpRNNTLoss(blank=blank)
        self.use_cuda = torch.cuda.is_available()

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Computes RNNT loss.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            inputs: A tuple where the first element is the unnormalized network
                :py:class:`torch.Tensor` outputs of size ``[batch, max_seq_len,
                max_output_seq_len + 1, vocab_size + 1)``. The second element
                is a Tuple of two :py:class:`torch.Tensor`s both of
                size ``[batch]`` that contain the lengths of a) the audio features
                logits and b) the target sequence logits.

            targets: A tuple where the first element is a
                :py:class:`torch.Tensor` such that each entry in the target
                sequence is a class index. Target indices cannot be the blank
                index. It must have size ``[batch, max_seq_len]``. In the former form each target
                sequence is padded to the length of the longest sequence and
                stacked.

                The second element is a :py:class:`torch.Tensor` that gives
                the lengths of the targets. Lengths are specified for each
                sequence to achieve masking under the assumption that sequences
                are padded to equal lengths.
        """
        logits, (logit_lens, _) = inputs
        y, y_lens = targets
        if self.use_cuda:
            logits = logits.cuda()
            logit_lens = logit_lens.cuda()
            y = y.cuda()
            y_lens = y_lens.cuda()

        if logit_lens.dtype != torch.int32:
            logit_lens = logit_lens.int()

        if y_lens.dtype != torch.int32:
            y_lens = y_lens.int()

        loss = self.rnnt_loss(
            acts=logits, labels=y, act_lens=logit_lens, label_lens=y_lens
        )

        return loss
