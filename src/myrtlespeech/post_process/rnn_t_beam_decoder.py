from collections import Counter
from collections import defaultdict
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import torch
from myrtlespeech.model.rnn_t import RNNT


class RNNTBeamDecoder(torch.nn.Module):
    """Decodes RNNT output using a beam search strategy.

    Args:
        blank_index: Index of the "blank" symbol. It is advised that the blank symbol
            is placed at the end of the alphabet in order to avoid different symbol
            index conventions in the prediction and joint networks (i.e. input and output of rnnt).
            However, this condition is not enforced here.

        model: A :py:class:`myrtlespeech.model.rnn_t.RNNT` model to use during decoding
            See the py:class:`myrtlespeech.model.rnn_t.RNNT` docstring for more information.

        length_norm: bool, default=False. If True, normalise log probabilities by length before the
            returning the "most probable" sequence. This avoids favouring short
            predictions and was used in the first Graves RNNT paper (2012): https://arxiv.org/pdf/1211.3711.pdf
            Default is False since the practice was discontinued by Graves in
            his later RNNT paper (2013): https://arxiv.org/pdf/1303.5778.pdf

        max_symbols_per_step: The maximum number of symbols that can be added to
            output sequence in a single time step. Default value is None: in this case
            the limit is set to 100 (to avoid the potentially infinite loop that
            could occur when with no limit).
        TODO beam_width
    """

    def __init__(
        self,
        blank_index: int,
        model: RNNT,
        length_norm: Optional[bool] = False,
        max_symbols_per_step: Optional[int] = None,
    ):
        if blank_index < 0:
            raise ValueError(f"blank_index={blank_index} must be >= 0")

        if max_symbols_per_step is None:
            max_symbols_per_step = 100  # i.e. to prevent infinite loop
        assert (
            max_symbols_per_step > 0
        ), "max_symbols_per_step must be a positive integer"

        super().__init__()
        self.blank_index = blank_index
        self.model = model
        self.length_norm = length_norm
        self.max_symbols_per_step = max_symbols_per_step
        raise NotImplementedError(
            "rnn_t_beam_decoder is not implemented. \
        Everything below this point is copied from CTC beam decoder"
        )

    """Decodes CTC output using a beam search.

    This is a reference implementation and its performance is *not* guaranteed
    to be useful for a production system.

    Based on the technique described in `Hannun et al. 2014
    <https://arxiv.org/pdf/1408.2873.pdf>`_.

    Args:
        blank_index: Index of the blank symbol in the ``alphabet_len``
            dimension of the ``x`` argument of
            :py:meth:`CTCBeamDecoder.forward`.

        beam_width: Width of the beam search. Must be a positive integer.

        max_symbols_per_step: The maximum number of symbols that can be added to
            output sequence in a single time step. Default value is None: in this case
            the limit is set to 1e2 (to avoid the potentially infinite loop).

        language_model: A ``Callable`` that takes a ``Tuple[int, ...]``
            representing (the indices of) a variable-length sequence of symbols
            and returns the probability of the last token (word) given all
            previous ones.

        lm_weight: Language model weight. Referred to as :math:`\alpha` in the
            paper above.

        separator_index: Index of the separator symbol in the ``alphabet_len``
            dimension of the ``x`` argument of
            :py:meth:`CTCBeamDecoder.forward`. This symbol is used to delineate
            tokens (words) in a predicted sequence. For example in English this
            is typically the index of the space: ``" "``. The ``language_model``,
            if set, is applied each time a token (word) is predicted.

        word_weight: Word count weight. Referred to as :math:`\beta` in the
            paper above.

            .. note::

                If ``separator_index is None`` this is not used (as there will
                only ever be one word!)

            .. note::

                The implementation differs from the paper as it adds 1 to the
                total word count (additive smoothing).

    Raises:
        :py:class:`ValueError`: If ``blank_index < 0``.

        :py:class:`ValueError`: If ``beam_width <= 0``.

        :py:class:`ValueError`:
            If ``prune_threshold < 0 or prune_threshold > 1.0``.

        :py:class:`ValueError`:
            If ``language_model is not None and lm_weight is None``.
    """

    def _n_words(self, prefix: List[int]) -> int:
        assert self.separator_index is not None
        n = 0
        prev = None
        for symbol in prefix:
            if symbol == self.separator_index and prev != self.separator_index:
                n += 1
            prev = symbol
        return n

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> List[List[int]]:
        r"""Decodes CTC output using a beam search.

        Args:
            x: A 3D :py:class:`torch.Tensor` of size ``(seq_len, batch,
                alphabet_len)`` where each entry contains the normalized
                probability of a given symbol in the alphabet at a given
                timestep for a specific batch element.

            lengths: A 1D :py:class:`torch.Tensor` giving the sequence length
                of each element in the batch. Must be an integer datatype.

        Returns:
            A ``List[List[int]]`` where the outer list contains ``batch``
            number of ``List[int]``\s. The :math:`i\textsuperscript{th}`
            ``List[int]`` corresponds to approximately the *most likely*
            sequence of symbols -- as found through a beam search -- for the
            :math:`i\textsuperscript{th}` sequence in ``x``.

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

        # each element in batch processed sequentially
        out: List[List[int]] = []
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
            A_prev: List[Tuple[int, ...]] = []
            A_prev.append(())

            ctc = x[:, i, :]
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
                                    and self.language_model is not None
                                    and c == self.separator_index
                                ):
                                    # keep mypy happy
                                    assert self.lm_weight is not None
                                    lm_prob = self.language_model(l_plus)
                                    p_l_plus *= lm_prob ** self.lm_weight

                                Pnb[t][l_plus] += p_l_plus

                            if l_plus not in A_prev:
                                # if not in beam then l_plus may have been
                                # discarded at the previous step and we can
                                # make use of this information for free
                                Pb[t][l_plus] += ctc[t][self.blank_index] * (
                                    Pb[t - 1][l_plus] + Pnb[t - 1][l_plus]
                                )
                                Pnb[t][l_plus] += ctc[t][c] * Pnb[t - 1][l_plus]

                # keep beam_width prefixes, scale scores by weighted number of
                # words to compensate for LM reducing scores (1 added to smooth
                # result)
                A_next = Pb[t] + Pnb[t]

                def sorter(l):
                    if self.separator_index is None:
                        return A_next[l]
                    return (
                        A_next[l] * (1 + self._n_words(l)) ** self.word_weight
                    )

                A_prev = sorted(A_next, key=sorter, reverse=True)
                A_prev = A_prev[: self.beam_width]

            out.append(list(A_prev[0]) if len(A_prev) > 0 else [])

        return out

    def extra_repr(self) -> str:
        return ",\n".join(
            [
                f"blank_index={self.blank_index}",
                f"beam_width={self.beam_width}",
                f"prune_threshold={self.prune_threshold}",
                f"language_model={self.language_model}",
                f"lm_weight={self.lm_weight}",
                f"separator_index={self.separator_index}",
                f"word_weight={self.word_weight}",
            ]
        )
