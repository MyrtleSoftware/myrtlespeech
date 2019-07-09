from typing import List

import torch


class CTCGreedyDecoder(torch.nn.Module):
    """Decodes CTC output using a greedy strategy.

    Args:
        blank_index: Index of the "blank" symbol.
    """

    def __init__(self, blank_index: int):
        super().__init__()
        self.blank_index = blank_index

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> List[List[int]]:
        r"""Decodes CTC output using a greedy strategy.

        Args:
            x: A 3D :py:class:`torch.Tensor` of size ``(seq_len, batch,
                alphabet_len)`` where each entry contains the normalized or
                unnormalized probability of a given symbol in the alphabet at a
                given timestep for a specific batch element.

            lengths: A 1D :py:class:`torch.Tensor` giving the sequence length
                of each element in the batch. Must be an integer datatype.

        Returns:
            A ``List[List[int]]`` where the outer list contains ``batch``
            number of ``List[int]``\s. The :math:`i\textsuperscript{th}`
            ``List[int]`` corresponds to approximately the *most likely*
            sequence of symbols -- as found by the greedy decoding strategy --
            for the :math:`i\textsuperscript{th}` sequence in ``x``.

        Raises:
            :py:class:`ValueError`: if ``lengths.dtype`` not in ``[torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]``.

            :py:class:`ValueError`: if the ``batch`` dimension of ``x`` is not
                equal to the ``len(lengths)``.

            :py:class:`ValueError`: if any of the sequence lengths in
                ``lengths`` are greater than the ``seq_len`` dimension of
                ``x``.
        """
        supported_dtypes = [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]
        if lengths.dtype not in supported_dtypes:
            raise ValueError(
                f"lengths.dtype={lengths.dtype} must be in {supported_dtypes}"
            )

        seq_len, x_batch, symbols = x.size()
        l_batch = len(lengths)
        if x_batch != l_batch:
            raise ValueError(
                f"batch size of x ({x_batch}) and lengths {l_batch} "
                "must be equal"
            )

        if not (lengths <= seq_len).all():
            raise ValueError(
                "length values must be less than or equal to x seq_len"
            )

        most_likely = x.argmax(dim=2)

        out = []
        for b in range(x_batch):
            sentence = []
            prev = None
            for i in range(lengths[b]):
                symb = most_likely[i, b].item()

                if symb != self.blank_index:
                    if prev is None or prev == self.blank_index or symb != prev:
                        sentence.append(symb)

                prev = symb
            out.append(sentence)

        return out
