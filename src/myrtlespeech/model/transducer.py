from typing import Tuple

import torch


class Transducer(torch.nn.Module):
    r"""`Transducer <https://arxiv.org/pdf/1211.3711.pdf>`_ Network.

    Args:
        encoder: A :py:class:`torch.nn.Module` to use as the transducer
            encoder. It must accept as input a Tuple where the first element
            is the audio feature sequence: a :py:class:`torch.Tensor` with
            size ``[batch, channels, features, max_input_seq_len]`` and the
            second element is a :py:class:`torch.Tensor` of size ``[batch]``
            where each entry represents the sequence length of the
            corresponding audio feature input.

            It must return a tuple where the first element is the result after
            applying the module to the audio input. It must have size
            ``[max_seq_len, batch, encoder_out_feat]``. The second element
            of the tuple return value is a :py:class:`torch.Tensor` with size
            ``[batch]`` where each entry represents the sequence length of the
            corresponding *output* sequence. These may be different than the
            input sequence lengths due to downsampling in `encoder`.

            It is possible **but not necessary** to use an initialized
            :py:class:`TransducerEncoder` class as `encoder`.

            Note that the encoder network is sometimes referred to as the
            'transcription network' in the literature.

        predict_net: A :py:class:`torch.nn.Module` to use as the transducer
            prediction network. It must accept as input a Tuple where the
            first element is the target label tensor with size ``[batch,
            max_label_length]`` and the second is a :py:class:`torch.Tensor`
            of size ``[batch]`` that contains the *input* lengths of these
            target label sequences.

            It must return a tuple where the first element is the result after
            applying the module to the input and it must have size ``[batch,
            max_label_length + 1, pred_net_out_feat]``. Note that the
            dimension at index 1 is ``max_label_length + 1`` since the
            start-of-sequence token is prepended to the label sequence.
            The second element of the returned Tuple is a
            :py:class:`torch.Tensor` of size ``[batch]`` with the containing
            the lengths of the target label sequence. These should be
            unchanged from the input lengths.

            It is possible **but not necessary** to use an initialized
            :py:class:`TransducerPredictNet` class as `predict_net`.

        joint_net: A :py:class:`torch.nn.Module` to use as the the transducer
            joint network. It must accept as input a Tuple where the first
            element is a :py:class:`torch.Tensor` with size
            ``[batch, max_seq_len, max_label_length + 1, hidden_dim]``
            where ``hidden_dim = encoder_out_feat + pred_net_out_feat`` as
            the ``encoder`` and ``predict_net`` ouputs are concatenated.
            The second element is a :py:class:`torch.Tensor` of size
            ``[batch]`` where each entry represents the sequence length of
            the ``encoder`` output sequences.

            It must return a tuple where the first element is the result after
            applying the module to the input. It must have size ``[batch,
            max_seq_len, max_label_length + 1, vocab_size + 1]``.
            ``max_seq_len`` is the length of the longest sequence in
            the batch that is output from ``encoder`` while
            ``max_label_length`` is the length of the longest *label*
            sequence in the batch that is output from ``predict_net``. Note
            that the dimension at index 2 is ``max_label_length + 1`` since
            the start-of-sequence label is prepended to the label sequence and
            the dimension at index 3 is ``vocab_size + 1`` because the blank
            symbol can be output.

            The second element is a :py:class:`torch.Tensor` of
            size ``[batch]`` where each entry represents the sequence length
            of the ``encoder`` features after ``joint_net`` has acted on them.

            It is possible **but not necessary** to use an initialized
            :py:class:`TransducerJointNet` class as `joint_net`.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        predict_net: torch.nn.Module,
        joint_net: torch.nn.Module,
    ):
        super().__init__()

        self.encode = encoder
        self.predict_net = predict_net
        self.joint_net = joint_net
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()

    def forward(
        self,
        x: Tuple[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns the result of applying the Transducer network to `x`.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            x: A Tuple where the first element is the input to `encoder`
                and the second element is the input to ``predict_net`` (see
                initialisation docstring).

        Returns:
            The output of ``joint_net``. See initialization docstring.
        """

        self._certify_inputs_forward(x)
        (x_inp, x_lens), (y, y_lens) = x

        if self.use_cuda:
            x_inp = x_inp.cuda()
            x_lens = x_lens.cuda()
            y = y.cuda()
            y_lens = y_lens.cuda()

        f = self.encode((x_inp, x_lens))  # f[0] = (T, B, H1)
        g = self.predict_net((y, y_lens))  # g[0] = (U, B, H2)

        out = self.joint_net((f, g))

        return out

    @staticmethod
    def _certify_inputs_forward(inp):
        try:
            (x, x_lens), (y, y_lens) = inp
        except ValueError as e:
            print(
                "Unable to unpack inputs to Transducer. Are you using the \
            `TransducerTraining()` callback \
            found in `myrtlespeech.run.callbacks.rnn_t_training`?"
            )
            raise e
        B1, C, I, T = x.shape
        B2, U = y.shape

        (B3,) = x_lens.shape
        (B4,) = y_lens.shape
        assert (
            B1 == B2 == B3 == B4
        ), "Batch size must be the same for inputs and targets"

        if not (x_lens <= T).all():
            raise ValueError(
                "x_lens must be less than or equal to max number of time-steps"
            )
        if not (y_lens <= U).all():
            raise ValueError(
                f"y_lens must be less than or equal to max number of output \
                symbols but {y_lens.max()} > {U}"
            )
