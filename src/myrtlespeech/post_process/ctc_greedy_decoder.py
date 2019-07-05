from typing import List

import torch


class CTCGreedyDecoder(torch.nn.Module):
    """

    """

    def __init__(self, blank_index):
        super().__init__()
        self.blank_index = blank_index

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> List[List[int]]:
        """

        Args:
            x: 3D (seq_len, batch, symbol probs (norm and unnorm both OK?))
            lengths: 1D (batch) of sequence lengths, must be int dtype

        Returns:
            List[List[int]]

        """
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
