import math

import torch

from myrtlespeech.post_process.rnn_t_decoder_base import TransducerDecoder


class RNNTGreedyDecoder(TransducerDecoder):
    """A greedy transducer decoder.

    Args:
        blank_symbol: See `Decoder`.
        model: Model to use for prediction.
        max_symbols_per_step: The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        cutoff_prob: Skip to next step in search if current highest character
            probability is less than this.
    """
    def __init__(self, blank_index, model, max_symbols_per_step=100,
                 cutoff_prob=0.01):
        super().__init__(blank_index, model)
        assert max_symbols_per_step is None or max_symbols_per_step > 0
        self.max_symbols = max_symbols_per_step
        self._log_cutoff_prob = math.log(cutoff_prob)

    def __call__(self, *inp):
        (x, y), (x_lens, y_lens) = inp
        return self.decode(x, x_lens)

    def decode(self, x, out_lens):
        """Returns a list of sentences given an input batch.

        Args:
            x: A tensor of size (batch, channels, features, seq_len)
                TODO was (seq_len, batch, in_features).
            out_lens: list of int representing the length of each sequence
                output sequence.

        Returns:
            list containing batch number of sentences (strings).
        """
        if self._model.is_cuda:
            x = x.cuda()
        if self._model.is_half:
            x = x.half()

        output = []
        for batch_idx in range(x.size(0)):
            inseq = x[batch_idx]
            channels, features, seq_len = inseq.shape
            inseq = inseq.view(channels*features, seq_len).T.unsqueeze(0)
            logitlen = out_lens[batch_idx]
            sentence = self._greedy_decode(inseq, logitlen)
            output.append(sentence)

        return output

    @torch.no_grad()
    def _greedy_decode(self, x, out_len):
        training_state = self._model.training
        self._model.eval()

        device = x.device

        fs = self._model.encode(x)

        hidden = None
        label = []
        for time_idx in range(out_len):
            f = fs[time_idx, :, :].unsqueeze(0)

            not_blank = True
            symbols_added = 0

            while not_blank and (
                    self.max_symbols is None or
                    symbols_added < self.max_symbols):
                g, hidden_prime = self._pred_step(
                    self._get_last_symb(label),
                    hidden,
                    device
                )
                logp = self._joint_step(f, g, log_normalize=False)[0, :]

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()

                if v < self._log_cutoff_prob:
                    break

                if k == self._blank_id:
                    not_blank = False
                else:
                    label.append(k)
                    hidden = hidden_prime
                symbols_added += 1

        self._model.train(training_state)
        return label
