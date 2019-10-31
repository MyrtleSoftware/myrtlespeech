from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from myrtlespeech.data.batch import collate_label_list
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
        # TODO: update this to support arbitrary cuda device idx
        self.device = "cuda:0" if self.model.use_cuda else "cpu"

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        lengths: Tuple[torch.Tensor, torch.Tensor],
    ) -> List[List[int]]:
        r"""Decodes RNNT output using a greedy strategy. Note that the input args are
        the same as the :py:class:`.RNNT` args but here the tuple of args must be unpacked
        with `.forward(*args)` while for the :py:class:`.RNNT` network they are passed
        as is: `.forward(args)`

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            inputs: A Tuple ``(x[0], x[1])``. ``x[0]`` is input to the network and is
                a Tuple ``x[0] = (x[0][0], x[0][1])`` where both elements are
                :py:class:`torch.Tensor`s. ``x[0][0]`` is the audio feature input
                with  size ``[batch, channels, features, max_input_seq_len]`` while ``x[0][1]`` is
                the target label tensor of size ``[batch, max_label_length]``.
                ``x[1]`` is a Tuple of two :py:class:`torch.Tensor`s both of
                size ``[batch]`` that contain the *input* lengths of a) the audio feature
                inputs ``x[1][0]`` and b) the target sequences ``x[1][1]``.
        Returns:
            A List of length Lists where each sublist contains the
        """

        # certify inputs and get dimensions
        dims = self.model._certify_inputs_forward((inputs, lengths))
        (batches, channel, audio_feat_input, max_seq_len, max_output_len) = dims

        audio_data, _ = self.model._prepare_inputs_forward(
            (inputs, lengths)
        )  # drop label_data

        out = []
        for b in range(batches):
            audio_len = audio_data[1][b].unsqueeze(0)
            audio_features = audio_data[0][b, :, :, :audio_len].unsqueeze(0)
            audio_inp = (audio_features, audio_len)

            sentence = self._greedy_decode(audio_inp)
            out.append(sentence)

        return out

    @torch.no_grad()
    def _greedy_decode(self, inp):
        training_state = self.model.training
        self.model.eval()

        # compute encoder:
        fs, fs_lens = self.model.encode(inp)
        fs = fs[: fs_lens[0], :, :]  # size: seq_len, batch = 1, rnn_features
        assert fs_lens[0] == fs.shape[0], "Time dimension comparison failed"

        hidden = None
        label = []

        for t in range(fs.shape[0]):

            f = fs[t, :, :].unsqueeze(0)
            # add length
            f = (f, torch.IntTensor([1]))
            not_blank = True
            symbols_added = 0
            while not_blank and symbols_added < self.max_symbols_per_step:

                g, hidden_prime = self._pred_step(
                    self._get_last_symb(label), hidden
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

        self.model.train(training_state)
        return label

    def _pred_step(self, label, hidden):
        if label == SOS:
            label_embedding = torch.zeros(
                (1, 1, self.model.dec_rnn.rnn.hidden_size), device=self.device
            )

            lengths = torch.IntTensor([1])  # i.e. length of target is 1

        else:
            if label > self.blank_index:
                label -= 1  # Since input label indexes will be offset by +1
                # for labels above blank. Avoiding this complexity
                # is the reason for using blank_index = (len(alphabet) - 1)

            collated = collate_label_list([[label]], device=self.device)
            label_embedding, lengths = self.model.embedding(collated)

        input = ((label_embedding, hidden), lengths)
        ((pred, hidden), pred_lens) = self.model.dec_rnn(input)

        return (pred, pred_lens), hidden

    def _joint_step(self, enc, pred):
        input = self.model._enc_pred_to_joint(enc, pred)

        logits, _ = self.model.joint(input)
        return torch.nn.functional.log_softmax(logits, dim=-1).squeeze()

    @staticmethod
    def _get_last_symb(labels):
        return SOS if labels == [] else labels[-1]


def log_aplusb(a, b):
    return max(a, b) + math.log1p(math.exp(-math.fabs(a - b)))
