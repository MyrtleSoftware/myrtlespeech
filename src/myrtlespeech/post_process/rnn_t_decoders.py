import math
from collections import Counter
from collections import defaultdict
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from myrtlespeech.model.rnn_t import RNNT
from myrtlespeech.post_process.rnn_t_decoder_base import RNNTDecoderBase


class RNNTGreedyDecoder(RNNTDecoderBase):
    """Decodes RNNT output using a greedy strategy.

    Args:
        See :py:class:`RNNTDecoderBase` for args.
    """

    def __init__(
        self,
        blank_index: int,
        model: RNNT,
        max_symbols_per_step: Optional[Union[int, None]] = None,
    ):

        super().__init__(
            blank_index=blank_index,
            model=model,
            max_symbols_per_step=max_symbols_per_step,
        )

    def decode(self, inp: Tuple[torch.Tensor, torch.Tensor]) -> List[int]:

        """Greedy RNNT decode method. See :py:class:`RNNTDecoderBase` for args"""

        fs, fs_lens = self.model.encode(inp)
        fs = fs[: fs_lens[0], :, :]  # size: seq_len, batch = 1, rnn_features
        assert fs_lens[0] == fs.shape[0], "Time dimension comparison failed"

        hidden = None
        label: List[int] = []

        for t in range(fs.shape[0]):

            f = fs[t, :, :].unsqueeze(0)

            # add length
            f = (f, torch.IntTensor([1]))

            not_blank = True
            symbols_added = 0
            while not_blank and symbols_added < self.max_symbols_per_step:

                g, hidden_prime = self._pred_step(
                    self._get_last_idx(label), hidden
                )
                logp = self._joint_step(f, g)

                # get index k, of max prob
                max_val, idx = logp.max(0)
                idx = idx.item()

                if idx == self.blank_index:
                    not_blank = False
                else:
                    label.append(idx)
                    hidden = hidden_prime
                symbols_added += 1

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


class RNNTBeamDecoder(RNNTDecoderBase):
    """Decodes RNNT output using a beam search strategy. This is a reference
    implementation and its performance is *not* guaranteed to be useful for a
    production system. Based on the technique described in `Graves 2012
    <https://arxiv.org/abs/1211.3711>`_.

    Args:

        beam_width: An int, default=4. The beam width for the decoding. Must be
            a positive integer.

        length_norm: bool, default=False. If True, normalise log probabilities by length before the
            returning the "most probable" sequence. This avoids favouring short
            predictions and was used in the first Graves RNNT paper (2012): https://arxiv.org/pdf/1211.3711.pdf
            Default is False since the practice was discontinued by Graves in
            his later RNNT paper (2013): https://arxiv.org/pdf/1303.5778.pdf

        See :py:class:`RNNTDecoderBase` for other args.

    """

    def __init__(
        self,
        blank_index: int,
        model: RNNT,
        beam_width: Optional[int] = 4,
        length_norm: Optional[bool] = False,
        max_symbols_per_step: Optional[int] = None,
    ):
        assert (
            isinstance(beam_width, int) and beam_width > 0
        ), "`beam_width` must be a positive integer"

        super().__init__(
            blank_index=blank_index,
            model=model,
            max_symbols_per_step=max_symbols_per_step,
        )

        self.beam_width = beam_width
        self.length_norm = length_norm

    def decode(self, inp: Tuple[torch.Tensor, torch.Tensor]) -> List[int]:

        """Beam RNNT decode method. See :py:class:`RNNTDecoderBase` for args"""

        fs, fs_lens = self.model.encode(inp)
        fs = fs[: fs_lens[0], :, :]  # size: seq_len, batch = 1, rnn_features
        assert fs_lens[0] == fs.shape[0], "Time dimension comparison failed"

        B = [Sequence(max_symbols=self.max_symbols_per_step)]
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
                len(B) < self.beam_width
                or B[self.beam_width - 1].logp < A[0].logp
            ):
                y_star = max(A, key=lambda a: (a.logp, len(a.labels)))
                A.remove(y_star)

                pred, hidden = self._pred_step(
                    self._get_last_idx(y_star.labels), y_star.h
                )
                logp = self._joint_step(f, pred)

                for k in range(logp.shape[0]):
                    yk = Sequence(y_star)
                    yk.logp += float(logp[k])
                    if k == self.blank_index:
                        if yk not in B:
                            yk.remaining_symbols = self.max_symbols_per_step
                            B.append(yk)
                        continue

                    yk.labels.append(k)
                    yk.times.append(t)
                    if self.max_symbols_per_step is not None:
                        yk.remaining_symbols -= 1
                        if yk.remaining_symbols < 0:
                            continue
                    if yk not in A:
                        yk.g.append(pred)
                        yk.h = hidden
                        A.append(yk)
                A.sort(key=lambda a: (-a.logp, len(a.labels)))
                B.sort(key=lambda a: (-a.logp, len(a.labels)))

            B = B[: self.beam_width]

        if self.length_norm:
            B.sort(key=lambda a: -a.logp / max(len(a.labels), 0.1))

        return B[0].labels
