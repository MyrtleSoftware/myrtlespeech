import math
from typing import List
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

        beam_width: The beam width for the decoding.

        length_norm: If :py:data`True`, log probabilities are normalised by
            length before the "most probable" sequence is returned. This
            avoids favouring of short predictions and was used in the first
            Graves `Transducer paper (2012)
            <https://arxiv.org/pdf/1211.3711.pdf>`_. Note that the practice
            was discontinued by Graves in his subsequent `2013 paper
            <https://arxiv.org/pdf/1303.5778.pdf>`_.

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
        beam_width: int = 4,
        length_norm: bool = False,
        max_symbols_per_step: int = 100,
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

        See :py:class:`TransducerDecoderBase` for args.
        """

        (fs, fs_lens), _ = self._model.encode(inp)
        fs = fs[
            : fs_lens.max(), :, :
        ]  # size: seq_len, batch = 1, rnn_features

        B = [Sequence()]
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
                        self._get_last_idx(y_hat.labels), y_hat.pred_hidden
                    )
                    idx = len(y_hat.labels)
                    logp = self._joint_step(f, pred)
                    curlogp = y_hat.logp + float(logp[y.labels[idx]])
                    for k in range(idx, len(y.labels) - 1):
                        logp = self._joint_step(f, y.pred_log[k])
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
                    self._get_last_idx(y_star.labels), y_star.pred_hidden
                )
                logp = self._joint_step(f, pred)

                for k in range(logp.shape[0]):
                    if logp[k] <= self._log_prune_threshold:
                        continue
                    yk = Sequence(y_star)
                    yk.logp += float(logp[k])
                    if k == self._blank_index:
                        if yk not in B:
                            yk.n_step_labels = 0
                            B.append(yk)
                        continue

                    yk.labels.append(k)
                    yk.times.append(t)
                    if yk.n_step_labels == self._max_symbols_per_step:
                        continue
                    yk.n_step_labels += 1
                    if yk not in A:
                        yk.pred_log.append(pred)
                        yk.pred_hidden = hidden
                        A.append(yk)
                A.sort(key=lambda a: (-a.logp, len(a.labels)))
                B.sort(key=lambda a: (-a.logp, len(a.labels)))
            B = B[: self._beam_width]

        if self._length_norm:
            B.sort(key=lambda a: -a.logp / max(len(a.labels), 0.1))

        label = B[0].labels
        del f, pred, hidden, logp, fs, B, A, y_star, yk
        return label

    def __repr__(self):
        str = super().__repr__()[:-1]
        str += f", beam_width={self._beam_width}"
        str += f", length_norm={self._length_norm}"
        return str + ")"


def log_aplusb(a, b):
    return max(a, b) + math.log1p(math.exp(-math.fabs(a - b)))


def is_prefix(a, b):
    """Returns True if a is a proper prefix of b."""
    return len(a) < len(b) and a == b[: len(a)]


class Sequence:
    """A candidate sequence in the beam search.

    Args:
        seq (Sequence): If not :py:data:`None`, an existing Sequence to
            initialize from. If not :py:data:`None`, ``pred_hidden`` must be
            :py:data:`None` else :py:data:`ValueError` is raised.

        pred_hidden: Predict net hidden state for the Sequence.

    Attributes:
        pred_log: A list containing the prediction network output
            :py:class:`torch.Tensor` s. ``pred_log[i]`` is the predict
            net output that was used when computing the ``i`` th symbol
            in ``labels``.

        labels: A List of ints containing the label indexes.

        times: A List of ints. ``times[i]`` is the time step at which
            ``labels[i]`` was added to the Sequence. Note that this does not
            necessarily correspond to the input time steps due to downsampling
            (pooling, convolutions, striding, etc).

        pred_hidden: Hidden state of the prediction network for the current
            sequence.

        logp: Log probability of the sequence. Initialised to ``math.log(1.0)``
            as all sequences should derive from a single starting sequence.

        n_step_labels: Number of labels added to ``labels`` during the current
            time step.
    """

    def __init__(self, seq=None, pred_hidden=None):
        if seq is None:
            self.pred_log = []
            self.labels = []
            self.times = []
            self.pred_hidden = pred_hidden
            self.logp = 0
            self.n_step_labels = 0
        else:
            if pred_hidden is not None:
                raise ValueError("pred_hidden must be None")
            self.pred_log = seq.pred_log[:]
            self.labels = seq.labels[:]
            self.times = seq.times[:]
            self.pred_hidden = seq.pred_hidden
            self.logp = seq.logp
            self.n_step_labels = seq.n_step_labels

    def __eq__(self, other):
        return self.labels == other.labels

    def __repr__(self):
        p = math.exp(self.logp)
        return (
            f"{self.__class__.__name__}(labels={self.labels}, "
            f"times={self.times}, "
            f"probability={p}, "
            f"n_step_labels={self.n_step_labels})"
        )
