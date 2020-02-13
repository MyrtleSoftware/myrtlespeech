from typing import Optional
from typing import Tuple

import torch
from myrtlespeech.model.rnn import RNNState
from myrtlespeech.run.callbacks.transducer_forward import TransducerForward


class Transducer(torch.nn.Module):
    r"""`Transducer <https://arxiv.org/pdf/1211.3711.pdf>`_ Network.

    Args:
        encoder: A :py:class:`torch.nn.Module` to use as the transducer
            encoder. It must accept as input a Tuple where the first element
            is the audio feature sequence: a :py:class:`torch.Tensor` with
            size ``[batch, channels, features, max_input_seq_len]`` and the
            second element is a :py:class:`torch.Tensor` of size ``[batch]``
            where each entry represents the sequence length of the
            corresponding audio feature input. It should also accept an
            Optional Arg ``hx`` representing the  ``RNNState`` of the encoder
            network.

            It must return a nested Tuple of the form ``((a, b), c)``
            where ``a`` is the result after applying the module to the audio
            input: a :py:class:`torch.Tensor` of size ``[max_seq_len, batch,
            encoder_out_feat]``. ``b`` is a :py:class:`torch.Tensor` with size
            ``[batch]`` where each entry represents the sequence length of the
            corresponding ``a`` sequence (these may be different to the
            input sequence lengths due to downsampling in ``encoder``). ``c``
            is the returned ``RNNState`` of ``encoder``.

            It is possible **but not necessary** to use an initialized
            :py:class:`TransducerEncoder` class as ``encoder``.

        predict_net: A :py:class:`torch.nn.Module` to use as the transducer
            prediction network. It must accept as input a Tuple where the
            first element is the target label  :py:class:`torch.Tensor` with
            size ``[batch, max_label_length]`` and the second is a
            :py:class:`torch.Tensor` of size ``[batch]`` that contains the
            *input* lengths of these target label sequences. It should also
            accept an Optional Arg ``hx`` representing the
            ``RNNState`` of the prediction network.

            It must return a nested Tuple of the form ``((a, b), c)``
            where ``a`` is the result after applying the module to the target
            inputs: a :py:class:`torch.Tensor` of size ``[batch,
            max_label_length + 1, pred_net_out_feat]`` (where the ``+ 1`` is
            results from the start-of-sequence token being prepended to the
            target label sequence). ``b`` is a :py:class:`torch.Tensor` with
            size ``[batch]`` where each entry represents the sequence length
            of the corresponding ``a`` sequence (these should be ``+ 1``
            greater than the input label sequence lengths.). ``c`` is the
            returned ``RNNState`` of ``predict_net``.

            It is possible **but not necessary** to use an initialized
            :py:class:`TransducerPredictNet` class as ``predict_net``.

        joint_net: A :py:class:`torch.nn.Module` to use as the the transducer
            joint network. It must accept as input a Tuple where the first
            element is the first element of the ``encoder`` Tuple output and
            the second is the first element of the ``predict_net`` Tuple
            output.

            It must return a nested Tuple of the form ``(a, b)``
            where ``a`` is the result after applying the module to the inputs:
            a :py:class:`torch.Tensor` of size ``[batch, max_seq_len,
            max_label_length + 1, vocab_size + 1]``. ``max_seq_len`` is the
            length of the longest sequence in the batch that is output from
            ``encoder`` while ``max_label_length`` is the length of the
            longest **label** sequence in the batch that is output from
            ``predict_net``. The ``+1`` in the ``max_label_length`` dimension
            is the result of start-of-symbol preprending (see ``predict_net``)
            and the ``+1`` in the ``vocab_size`` dimension indicates that the
            blank symbol is a valid ``joint_net`` output. ``b`` is a
            :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represents the sequence length of the ``encoder`` features
            after ``joint_net`` has acted on them.

            It is possible **but not necessary** to use an initialized
            :py:class:`TransducerJointNet` class as ``joint_net``.

    Attributes:
        callbacks: A collection of :py:class:`Callback`\s that will be
            automatically added to the :py:class:`CallbackHandler` if an
            instance of this class is passed as the ``model`` argument.
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
        self.callbacks = self._get_callbacks()
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()

    def forward(
        self,
        x: Tuple[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ],
        hx_enc: Optional[RNNState] = None,
        hx_pred: Optional[RNNState] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[RNNState, RNNState]]:
        r"""Returns the result of applying the Transducer network to ``x``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            x: A Tuple where the first element is the input to `encoder`
                and the second element is the input to ``predict_net`` (see
                initialisation docstring).

            hx_enc: Optional ``RNNState`` of the encoder network.

            hx_pred: Optional ``RNNState`` of the encoder network.

        Returns:
            A Tuple of the form ``((a, b), (c, d))`` where ``(a, b)`` is
            the ``joint_net`` output (see initialization docstring) and ``c``
            and ``d`` are the returned ``RNNState``s of the ``encoder`` and
            ``predict_net`` respectively.
        """
        self._certify_inputs_forward(x)
        (x_inp, x_lens), (y, y_lens) = x

        if self.use_cuda:
            x_inp = x_inp.cuda()
            x_lens = x_lens.cuda()
            y = y.cuda()
            y_lens = y_lens.cuda()

        f, hx_f = self.encode((x_inp, x_lens), hx=hx_enc)
        g, hx_g = self.predict_net((y, y_lens), hx=hx_pred)
        h = self.joint_net((f, g))

        return h, (hx_f, hx_g)

    @staticmethod
    def _certify_inputs_forward(inp):
        try:
            (x, x_lens), (y, y_lens) = inp
        except ValueError as e:
            print(
                "Unable to unpack inputs to Transducer. Are you passing \
            the Transducer module to the CallbackHandler as `model` arg?"
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

    def _get_callbacks(self):
        """Returns a list of required callbacks."""
        return [TransducerForward()]
