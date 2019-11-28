import math
from typing import List
from typing import Optional
from typing import Tuple

import torch
from myrtlespeech.model.transducer import Transducer
from myrtlespeech.post_process.transducer_decoder_base import (
    TransducerDecoderBase,
)


class TransducerBeamDecoder(TransducerDecoderBase):
    """Decodes Transducer output using a beam search strategy.

    This is a reference implementation and its performance is *not* guaranteed
    to be useful for a production system. Based on the technique described in
    `Graves 2012 <https://arxiv.org/abs/1211.3711>`_.

    Args:
        blank_index: See :py:class:`TransducerDecoderBase`.

        model: See :py:class:`TransducerDecoderBase`.

        beam_width: An int, default=4. The beam width for the decoding. Must be
            a positive integer.

        length_norm: bool, default=False. If True, normalise log probabilities
            by length before the returning the "most probable" sequence. This
            avoids favouring short predictions and was used in the first Graves
            Transducer paper (2012): https://arxiv.org/pdf/1211.3711.pdf.
            Default is False since the practice was discontinued by Graves in
            his subsequent paper (2013): https://arxiv.org/pdf/1303.5778.pdf

        max_symbols_per_step: See :py:class:`TransducerDecoderBase`.

        prune_threshold: Affects the list of symbols to consider when extending
            a prefix at each step. A prefix is not extended with a given symbol
            and added to the beam if the symbol has probability less than this
            threshold.
    """

    def __init__(
        self,
        blank_index: int,
        model: Transducer,
        beam_width: Optional[int] = 4,
        length_norm: Optional[bool] = False,
        max_symbols_per_step: Optional[int] = None,
        prune_threshold: float = 0.001,
    ):
        assert (
            isinstance(beam_width, int) and beam_width > 0
        ), "`beam_width` must be a positive integer"

        super().__init__(
            blank_index=blank_index,
            model=model,
            max_symbols_per_step=max_symbols_per_step,
        )

        self._beam_width = beam_width
        self._length_norm = length_norm
        self._log_prune_threshold = math.log(prune_threshold)

    def decode(self, inp: Tuple[torch.Tensor, torch.Tensor]) -> List[int]:
        r"""Beam Transducer decode method.

        Args:
            See :py:class:`TransducerDecoderBase`.

        Returns:
            See :py:class:`TransducerDecoderBase`.
        """

        fs, fs_lens = self._model.encode(inp)
        fs = fs[: fs_lens[0], :, :]  # size: seq_len, batch = 1, rnn_features
        assert (
            fs_lens[0] == fs.shape[0]
        ), f"Time dimension comparison failed. {fs_lens[0]} != {fs.shape[0]}"

        B = [Sequence(max_symbols=self._max_symbols_per_step)]
        for t in range(fs.shape[0]):
            f = fs[t, :, :].unsqueeze(0)
            # add length
            f = (f, torch.IntTensor([1]))

            A = sorted(B, key=lambda a: len(a.labels), reverse=True)
            B = []

            # 1) prefix step
            for i, y in enumerate(A):
                for j in range(i + 1, len(A)):
                    y_hat = A[j]
                    if not is_prefix(y_hat.labels, y.labels):
                        continue
                    pred, _ = self._pred_step(
                        self._get_last_idx(y_hat.labels), y_hat.h
                    )
                    idx = len(y_hat.labels)
                    logp = self._joint_step(f, pred)
                    curlogp = y_hat.logp + float(logp[y.labels[idx]])
                    for k in range(idx, len(y.labels) - 1):
                        logp = self._joint_step(f, y.g[k])
                        curlogp += float(logp[y.labels[k + 1]])
                    y.logp = log_aplusb(y.logp, curlogp)

            A.sort(key=lambda a: (-a.logp, len(a.labels)))

            # 2) main beam search
            while len(A) > 0 and (
                len(B) < self._beam_width
                or B[self._beam_width - 1].logp < A[0].logp
            ):
                y_star = max(A, key=lambda a: (a.logp, len(a.labels)))
                A.remove(y_star)

                pred, hidden = self._pred_step(
                    self._get_last_idx(y_star.labels), y_star.h
                )
                logp = self._joint_step(f, pred)

                for k in range(logp.shape[0]):
                    if logp[k] <= self._log_prune_threshold:
                        continue
                    yk = Sequence(y_star)
                    yk.logp += float(logp[k])
                    if k == self._blank_index:
                        if yk not in B:
                            yk.remaining_symbols = self._max_symbols_per_step
                            B.append(yk)
                        continue

                    yk.labels.append(k)
                    yk.times.append(t)
                    if self._max_symbols_per_step is not None:
                        yk.remaining_symbols -= 1
                        if yk.remaining_symbols < 0:
                            continue
                    if yk not in A:
                        yk.g.append(pred)
                        yk.h = hidden
                        A.append(yk)
                A.sort(key=lambda a: (-a.logp, len(a.labels)))
                B.sort(key=lambda a: (-a.logp, len(a.labels)))
            B = B[: self._beam_width]

        if self._length_norm:
            B.sort(key=lambda a: -a.logp / max(len(a.labels), 0.1))

        label = B[0].labels
        del f, pred, hidden, logp, fs, B, A, y_star, yk
        return label


def log_aplusb(a, b):
    return max(a, b) + math.log1p(math.exp(-math.fabs(a - b)))


def is_prefix(a, b):
    if a == b or len(a) >= len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


class Sequence:
    def __init__(self, seq=None, hidden=None, max_symbols=None):
        if seq is None:
            self.g = []  # predictions of phoneme language model
            self.labels = []  # prediction phoneme label
            self.times = (
                []
            )  # list of timesteps at which predictions are emmitted
            self.h = hidden
            self.logp = 0  # probability of this sequence, in log scale
            self.remaining_symbols = max_symbols
        else:
            self.g = seq.g[:]  # save for prefixsum
            self.labels = seq.labels[:]
            self.times = seq.times[:]
            self.h = seq.h
            self.logp = seq.logp
            self.remaining_symbols = seq.remaining_symbols

    def __eq__(self, other):
        return self.labels == other.labels

    def __str__(self):
        return f"{self.labels, self.logp}"
