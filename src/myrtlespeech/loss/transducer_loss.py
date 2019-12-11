from typing import Tuple

import torch
from warprnnt_pytorch import RNNTLoss as WarpTransducerLoss


class TransducerLoss(torch.nn.Module):
    """Wrapped :py:class:`warprnnt_pytorch.RNNTLoss`.

    Args:
        blank: Index of the blank label.

        reduction: A string that specifies the reduction to apply to the
            output. It can take the following values:

                none:
                    No reduction will be applied.

                mean:
                    The output losses will be divided by the target lengths and
                    then the mean over the batch is taken.

                sum:
                    Sum all losses in a batch.

    Attributes:
        transducer_loss: A :py:class:`warprnnt_pytorch.RNNTLoss` instance.
        use_cuda: If true, loss is evaluated on GPU.
    """

    def __init__(self, blank: int, reduction: str = "mean"):
        super().__init__()
        self.transducer_loss = WarpTransducerLoss(
            blank=blank, reduction=reduction
        )
        self.use_cuda = torch.cuda.is_available()

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Computes Transducer loss.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            inputs: A Tuple where the first element is the unnormalized output
                of the :py:class:`Transducer` network: a
                :py:class:`torch.Tensor` with size ``[batch, max_seq_len,
                max_label_length + 1, vocab_size + 1]``. ``max_seq_len`` is the
                length of the longest sequence in the batch output from the
                :py:attr:`Transducer.encoder` network whereas
                ``max_label_seq_len`` is the length of the longest *label*
                sequence in the batch output from the
                :py:attr:`Transducer.prediction`. Note that the dimension at
                index 2 is ``max_label_seq_len + 1`` since the
                start-of-sequence label is prepended to the label sequence and
                the dimension at index 3 is ``vocab_size + 1`` because the
                blank symbol can be output.

                The second element is a :py:class:`torch.Tensor` of size
                ``[batch]`` that contains the :py:attr:`Transducer.encoder`
                sequence output lengths.

            targets: A tuple where the first element is a
                :py:class:`torch.Tensor` such that each entry in the target
                sequence is a class index. Target indices cannot be the blank
                index. It must have size ``[batch, max_seq_len]``. In the
                former form each target sequence is padded to the length of
                the longest sequence and stacked.

                The second element is a :py:class:`torch.Tensor` that gives
                the lengths of the targets. Lengths are specified for each
                sequence to achieve masking under the assumption that sequences
                are padded to equal lengths.
        """

        logits, logit_lens = inputs
        y, y_lens = targets

        # cast to required types
        if logits.dtype != torch.float:
            logits = logits.float()

        if y.dtype != torch.int32:
            y = y.int()

        if logit_lens.dtype != torch.int32:
            logit_lens = logit_lens.int()

        if y_lens.dtype != torch.int32:
            y_lens = y_lens.int()

        # send to gpu
        if self.use_cuda:
            logits = logits.cuda()
            logit_lens = logit_lens.cuda()
            y = y.cuda()
            y_lens = y_lens.cuda()

        loss = self.transducer_loss(
            acts=logits, labels=y, act_lens=logit_lens, label_lens=y_lens
        )

        # del new variables that may have been created due to float/int/cuda()
        del logits, y, logit_lens, y_lens, inputs, targets

        return loss
