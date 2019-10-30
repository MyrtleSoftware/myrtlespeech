from typing import List
from typing import Optional
from typing import Union

import torch
from myrtlespeech.model.rnn_t import RNNT

SOS = -1  # Start of sequence


class RNNTGreedyDecoder(torch.nn.Module):
    """Decodes RNNT output using a greedy strategy.

    Args:
        blank_index: Index of the "blank" symbol. It is advised that the blank symbol
            is placed at the end of the alphabet in order to avoid different symbol
            index conventions in the prediction and joint networks (i.e. input and output of rnnt).
            However, this condition is not enforced here.

        model: A :py:class:`myrtlespeech.model.rnn_t.RNNT` model to use during decoding
            See the py:class:`myrtlespeech.model.rnn_t.RNNT` docstring for more information.

        max_symbols_per_step: The maximum number of symbols that can be added to
            output sequence in a single time step. Default value is None: in this case
            the limit is set to 100 (to avoid the potentially infinite loop that
            could occur when with no limit).
    """

    def __init__(
        self,
        blank_index: int,
        model: RNNT,
        max_symbols_per_step: Optional[Union[int, None]] = None,
    ):
        if blank_index < 0:
            raise ValueError(f"blank_index={blank_index} must be >= 0")

        if max_symbols_per_step is None:
            max_symbols_per_step = 100  # i.e. to prevent infinite loop
        assert max_symbols_per_step is not None
        assert (
            max_symbols_per_step > 0
        ), "max_symbols_per_step must be a positive integer"

        super().__init__()
        self.blank_index = blank_index
        self.model = model
        self.max_symbols_per_step = max_symbols_per_step

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> List[List[int]]:
        r"""Decodes RNNT output using a greedy strategy.

        TODO
        """

        audio_seq_len, x_batch, symbols = x.size()
        l_batch = len(lengths)
        if x_batch != l_batch:
            raise ValueError(
                f"batch size of x ({x_batch}) and lengths {l_batch} "
                "must be equal"
            )

        if not (lengths <= audio_seq_len).all():
            raise ValueError(
                "length values must be less than or equal to x audio_seq_len"
            )

        out = []
        for b in range(x_batch):
            inseq = x[: lengths[b], b, :].unsqueeze(1)
            inp = (inseq, lengths[b])
            sentence = self._greedy_decode(inp)
            out.append(sentence)

        return out

    @torch.no_grad()
    def _greedy_decode(self, inp):
        training_state = self.model.training
        self.model.eval()

        device = "cuda:0" if self.model.is_cuda else "cpu"

        indata, length = inp

        indata = indata.to(device)
        length = length.to(device)

        fs, fs_lens = self.model.encode((indata, length))
        fs = fs[:, :fs_lens, :]
        hidden = None
        label = []

        for t in range(fs.shape[0]):

            f = fs[:, t, :].unsqueeze(1)

            not_blank = True
            symbols_added = 0
            while not_blank and symbols_added < self.max_symbols_per_step:

                g, hidden_prime = self._pred_step(
                    self._get_last_symb(label), hidden
                )
                logp = self._joint_step(f, g)

                # get index k, of max prob
                v, idx = logp.max(0)
                idx = idx.item()

                if idx == self.blank_index:
                    not_blank = False
                else:
                    label.append(idx)
                    hidden = hidden_prime
                symbols_added += 1

        self.model.train(training_state)
        return label

    def _pred_step(self, label, hidden):
        if label == SOS:
            label_embedding = torch.zeros(
                (1, 1, self.model.dec_rnn.rnn.hidden_size), device=device
            )

        else:
            if label > self.blank_index:
                label -= 1
            collated = label_collate([[label]]).to(device)
            label_embedding = self.model.embedding(collated)

        pred, hidden = self.model.dec_rnn(label_embedding, hidden)

        return pred, hidden

    def _joint_step(self, enc, pred):
        logits = self.model.joint(enc, pred)
        return torch.nn.log_softmax(logits, dim=-1).squeeze()

    @staticmethod
    def _get_last_symb(labels):
        return SOS if labels == [] else labels[-1]


def log_aplusb(a, b):
    return max(a, b) + math.log1p(math.exp(-math.fabs(a - b)))
