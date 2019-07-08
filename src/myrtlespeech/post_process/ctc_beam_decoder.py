from typing import Callable, Tuple, List, Optional, Union
from collections import defaultdict, Counter

import torch


class CTCBeamDecoder(torch.nn.Module):
    """Decodes CTC output using a beam search.

    This is a reference implementation and its performance is *not* guaranteed
    to be useful for a production system.

    Args:
        blank_index: Index of the "blank" symbol.
        beam_width: TODO
        prune_threshold: TODO
        separator_index: TODO
        language_model: Takes Tuple[int] representing sentence, returns
            probability of last word given all previous words
    """

    def __init__(
        self,
        blank_index: int,
        beam_width: int,
        prune_threshold: float = 0.001,
        separator_index: Optional[int] = None,
        language_model: Optional[Callable[[Tuple[int]], float]] = None,
    ):
        super().__init__()
        self.blank_index = blank_index
        self.beam_width = beam_width
        self.prune_threshold = prune_threshold
        self.separator_index = separator_index
        self.language_model = language_model

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> List[List[int]]:
        r"""

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
            sequence of symbols -- as found through a beam search -- for the
            :math:`i\textsuperscript{th}` sequence in ``x``.
        """
        seq_len, x_batch, symbols = x.size()

        # each element in batch processed sequentially
        for i in range(x_batch):
            # Pb[t][l]: an estimate of the probability of *prefix* l, at step
            #           t, based on paths ending in a blank that map to l
            #
            # Pnb[t][l]: an estimate of the probability of *prefix* l, at step
            #            t, based on paths that do not end in a blank and map
            #            to l
            Pb, Pnb = defaultdict(Counter), defaultdict(Counter)  # type: ignore
            # prob of null sequence ending in a blank and non-blank after
            # seeing no input (-1 as the first time step seen has index 0)
            Pb[-1][()] = 1.0
            Pnb[-1][()] = 0.0
            # previous beam, init to only null sequence
            A_prev: List[Tuple[int]] = []
            A_prev.append(())  # type: ignore

            ctc = x[:, i, :]
            out: List[List[int]] = []
            for t in range(lengths[i]):
                for l in A_prev:
                    for c in range(symbols):
                        if ctc[t][c] <= self.prune_threshold:
                            continue

                        if c == self.blank_index:
                            # prefix stays same, update probability estimate
                            Pb[t][l] += ctc[t][c] * (
                                Pb[t - 1][l] + Pnb[t - 1][l]
                            )
                        else:
                            # extending prefix by non-blank
                            l_plus = l + (c,)

                            if len(l) > 0 and c == l[-1]:
                                # extending by same non-blank end symbol
                                Pnb[t][l_plus] += ctc[t][c] * Pb[t - 1][l]
                                Pnb[t][l] += ctc[t][c] * Pnb[t - 1][l]
                            else:
                                # extending by different non-blank end symbol
                                p_l_plus = ctc[t][c] * (
                                    Pb[t - 1][l] + Pnb[t - 1][l]
                                )

                                if (
                                    self.separator_index is not None
                                    and c == self.separator_index
                                    and self.language_model is not None
                                ):
                                    # include LM symbol is separator and LM
                                    raise ValueError("todo: support LM")
                                    lm_prob = self.language_model(l_plus)
                                    p_l_plus *= lm_prob ** self.alpha

                                Pnb[t][l_plus] += p_l_plus

                            if l_plus not in A_prev:
                                # if not in beam then l_plus may have been
                                # discarded at the previous step and we can
                                # make use of this information for free
                                Pb[t][l_plus] += ctc[t][self.blank_index] * (
                                    Pb[t - 1][l_plus] + Pnb[t - 1][l_plus]
                                )
                                Pnb[t][l_plus] += ctc[t][c] * Pnb[t - 1][l_plus]

                # keep beam_width prefixes
                print(Pb[t] + Pnb[t])
                A_next = Pb[t] + Pnb[t]
                A_prev = sorted(A_next, key=A_next.get, reverse=True)
                A_prev = A_prev[: self.beam_width]

            out.append(list(A_prev[0]) if len(A_prev) > 0 else [])

        return out
