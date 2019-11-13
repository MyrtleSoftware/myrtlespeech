from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from myrtlespeech.data.batch import collate_label_list
from myrtlespeech.model.rnn_t import RNNT

SOS = -1  # Start of sequence


class RNNTDecoderBase(torch.nn.Module):
    r"""Base RNNT decoder class.

    *This should not be instantiated directly.* Instead use specific
    decoders (e.g. :py:class:`RNNTGreedyDecoder` or
    :py:class:`RNNTBeamDecoder`).

    Args:
        blank_index: Index of the "blank" symbol. It is advised that the blank
            symbol is placed at the end of the alphabet in order to avoid
            different symbol index conventions in the prediction and joint
            networks (i.e. input and output of RNN-T) but this condition is not
            enforced here.

        model: A :py:class:`myrtlespeech.model.rnn_t.RNNT` model to use during
            decoding. See the py:class:`myrtlespeech.model.rnn_t.RNNT`
            docstring for more information.

        max_symbols_per_step: The maximum number of symbols that can be added to
            output sequence in a single time step. Default value is None: in
            this case the limit is set to 100 (to avoid the potentially
            infinite loop that could occur with no limit).

    Properties:
        _pred_step(label, hidden): performs a single step of prediction network.
        _joint_step(enc, pred): performs a single step of the joint network.
        _get_last_idx(label): gets final index of a list of indexes.
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

        # TODO: update this to support arbitrary cuda device idx:
        self.device = "cuda:0" if self.model.use_cuda else "cpu"

    @torch.no_grad()
    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        lengths: Tuple[torch.Tensor, torch.Tensor],
    ) -> List[List[int]]:
        r"""Decodes RNN-T output.

        Note that the input args are the same as the :py:class:`.RNNT` args but
        here the tuple of args is unpacked with `.forward(*args)` while
        for the :py:class:`.RNNT` network they are passed as is:
        `.forward(args)`.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            inputs: A Tuple of inputs to the network where both elements are
                :py:class:`torch.Tensor`s. ``inputs[0]`` is the audio
                feature input with  size ``[batch, channels, features,
                max_input_seq_len]`` while ``inputs[1]`` is the target label
                tensor of size ``[batch, max_label_length]``.

            lengths: A Tuple of two :py:class:`torch.Tensor`s both of
                size ``[batch]`` that contain the *input* lengths of a) the
                audio feature inputs ``lengths[0]`` and b) the target sequences
                ``lengths[1]``.

        Returns:
            A List of Lists where each sublist contains the index predictions
                of the decoder.
        """
        training_state = self.model.training
        self.model.eval()

        # certify inputs and get dimensions
        (batches, _, _, _, _) = self.model._certify_inputs_forward(
            (inputs, lengths)
        )

        audio_data, label_data = self.model._prepare_inputs_forward(
            (inputs, lengths), self.model.use_cuda
        )

        # since label_data *should not* be used in decoding, delete it
        # explicitly here:
        del label_data

        preds = []
        for b in range(batches):
            audio_len = audio_data[1][b].unsqueeze(0)
            audio_features = audio_data[0][b, :, :, :audio_len].unsqueeze(0)
            audio_inp = (audio_features, audio_len)
            sentence = self.decode(audio_inp)
            preds.append(sentence)

        # restore training state
        self.model.train(training_state)

        del audio_inp, audio_features, audio_data, audio_len, inputs, lengths
        return preds

    def decode(self, inp: Tuple[torch.Tensor, torch.Tensor]) -> List[int]:
        r"""Decodes a single sample.

        Args:
            inp: Tuple where the first element is the encoder
                input (a :py:`torch.Tensor`) with size ``[batch, channels,
                features, max_input_seq_len]`` and the second element is a
                :py:class:`torch.Tensor` of size ``[batch]`` where each entry
                represents the sequence length of the corresponding *input*
                sequence to the rnn.  Note that `inp` is passed straight to
                :py:class:`myrtlespeech.model.rnn_t.RNNTEncoder`.

        Returns:
            A List of length `[batch]` where each element is a List of indexes.

        """
        raise NotImplementedError(
            "decode method not implemented. Do not \
            instantiate `RNNTDecoderBase` directly. Instead use a specific \
            e.g. `RNNTGreedyDecoder`"
        )

    def _pred_step(self, label, hidden):
        b"""Performs a step of the model prediction network during inference.
        """
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
            del collated

        inp = ((label_embedding, hidden), lengths)
        ((pred, hidden), pred_lens) = self.model.dec_rnn(inp)
        del label_embedding, inp, lengths
        return (pred, pred_lens), hidden

    def _joint_step(self, enc, pred):
        r"""Performs a step of the model joint network during inference.
        """
        input = self.model._enc_pred_to_joint(enc, pred)

        logits, _ = self.model.joint(input)
        res = torch.nn.functional.log_softmax(logits, dim=-1).squeeze()
        assert (
            len(res.shape) == 1
        ), "this _joint_step result should have just one non-zero dimension"

        del logits, input
        return res

    def _get_last_idx(self, labels):
        b"""Returns the final index of a list of labels."""
        return SOS if labels == [] else labels[-1]

    def __repr__(self):
        string = self._get_name() + "("
        string += f"max_symbols_per_step={self.max_symbols_per_step}, "
        string += f"blank_index={self.blank_index}"
        if hasattr(self, "beam_width"):
            string += f", beam_width={self.beam_width}"
        string += ")"
        return string
