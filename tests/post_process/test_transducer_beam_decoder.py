from typing import Tuple

import torch
from myrtlespeech.model.rnn_t import RNNTEncoder
from myrtlespeech.model.rnn_t import RNNTJointNet
from myrtlespeech.model.rnn_t import RNNTPredictNet
from myrtlespeech.model.transducer import Transducer
from myrtlespeech.post_process.transducer_beam_decoder import (
    TransducerBeamDecoder,
)


# Fixtures and Strategies -----------------------------------------------------


class DummyPredictNet(RNNTPredictNet):
    """An :py:class:`RNNTPredictNet` with overridden forward method.

    Note that ``embedding=None`` will be passed to :py:meth:`super.__init__`
    as the :py:meth:`embed` method that calls forward on
    :py:class:`embedding` in :py:class:`RNNTPredictNet` is overridden below.

    Args:
        pred_nn: module with same :py:meth:`__call__` API as `pred_nn` in
            :py:class:`RNNTPredictNet`.
    """

    def __init__(self, pred_nn):
        super().__init__(embedding=None, pred_nn=pred_nn)

    def embed(self, x):
        """Overridden `:py:meth:`embed`."""
        res = torch.tensor([1, x[0].item() + 1]).unsqueeze(0).unsqueeze(0)
        x = res, x[1]

        return x


class PredNN:
    """Class to override `pred_nn` in :py:meth:`RNNTPredictNet`.

    This class replicates the :py:class:`RNNTPredictNet.pred_nn` API
    (which in turn is the same as an :py:class:`RNN` with `batch_first=True`).

    Args:
        hidden_size: fake `hidden_size` of module.
    """

    def __init__(self, hidden_size=3):
        self.hidden_size = hidden_size

    def __call__(self, x):
        """Replicates `pred_nn` :py:meth:`forward` method.

        This works by using the hidden state to decide which characters to
        upweight at a given timestep.

        Args:
            x: See :py:class:`RNN` with `batch_first=True`.

        Returns:
            x: See :py:class:`RNN` with `batch_first=True`.
        """

        if isinstance(x[0], torch.Tensor):
            embedded = x[0]
            state = None
            return_tuple = False
        elif isinstance(x[0], tuple) and len(x[0]) == 2:
            embedded, state = x[0]
            return_tuple = True
        else:
            raise ValueError(
                "`x[0]` must be of form (input, hidden) or (input)."
            )

        B, U_, H = embedded.shape

        assert B == U_ and B == 1, "Currently only supports batch=seq len== 1"

        lengths = x[1]

        if state is None:
            # `state` is the index of char to upweight.
            # In first instance upweight character at index 1
            state = torch.IntTensor([1])
        if embedded.squeeze().int()[0] == 0:
            # i.e. if this is the SOS,
            # assign probability of 0.2 to all three chars:
            res = torch.ones(3, dtype=torch.float32) * 0.2
        else:
            res = torch.ones(3, dtype=torch.float32) * 0.1
            state_to_upweight = embedded.squeeze().int()[1]  # 0 or 1
            res[state_to_upweight] += 0.3
        res[state.item()] += 0.4
        out, hid = torch.log(res), (state + 1) % 3
        # blow up to full dimension
        out = out.unsqueeze(0).unsqueeze(0)

        if return_tuple:
            return (out, hid), lengths
        else:
            return out, lengths


class DummyTransducerEncoder(RNNTEncoder):
    r"""An :py:class:`RNNTEncoder` with overridden forward method.

    This is necessary because in this worked example the audio inputs are
    simply returned as-is (with the dimension change required by the
    :py:class:`Transducer` API).
    """

    def __init__(self):
        rnn1_not_used_except_in_init = torch.nn.RNN(2, 2)
        super().__init__(rnn1=rnn1_not_used_except_in_init)

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replicates :py:attr:`Transducer.encoder` API.

        Args:
            See :py:attr:`Transducer.encoder`.
        Returns:
            See :py:attr:`Transducer.encoder`.
        """

        h = x[0]  # B, C, H, T
        B, C, F, T = h.shape

        assert (
            C == 1
        ), f"In DummyTransducerModel(), input channels must == 1 but C == {C}"
        h = h.squeeze(1)  # B, H, T
        h = h.permute(2, 0, 1)

        return h, x[1]


class DummyJointNet(RNNTJointNet):
    """A :py:class:`RNNTJointNet` with overridden forward method.

    This is necessary as :py:class:`RNNTJointNet` concatenates the encoder
    and decoder output, which breaks this worked example.
    """

    def __init__(self):
        super().__init__(fc=None)

    def forward(self, x):
        """Replicates :py:attr:`Transducer.joint_net` API.

        Args:
            See :py:attr:`Transducer.joint_net`.
        Returns:
            See :py:attr:`Transducer.joint_net`.
        """
        (f, f_lens), (g, g_lens) = x
        T, B1, H1 = f.shape
        B2, U_, H2 = g.shape

        assert (
            B1 == B2
        ), "Batch size from prediction network and transcription must be equal"

        f = f.transpose(1, 0)  # (T, B, H1) -> (B, T, H1)
        f = f.unsqueeze(dim=2)  # (B, T, 1, H)

        g = g.unsqueeze(dim=1)  # (B, 1, U_, H)

        return (f + g), f_lens


class TransducerBeamDecoderDummy(TransducerBeamDecoder):
    """A Decoder class which overrides :py:meth:`_joint_step` method."""

    def _joint_step(self, enc, pred):
        """Overrrides :py:meth:`_joint_step()`.

        This is necessary as :py:meth:`TransducerDecoderBase._joint_step()`
        performs a :py:meth:`log_softmax` which breaks this worked example.
        """
        logits, _ = self._model.joint_net((enc, pred))
        return logits.squeeze()


def get_dummy_transducer(hidden_size=3):
    encoder = DummyTransducerEncoder()
    predict_net = DummyPredictNet(pred_nn=PredNN(hidden_size=hidden_size))
    joint_net = DummyJointNet()
    model = Transducer(
        encoder=encoder, predict_net=predict_net, joint_net=joint_net
    )
    model.eval()
    return model


def get_fixed_decoder(max_symbols_per_step=100):
    # alphabet = ["_", "a", "b"]
    blank_index = 0
    model = get_dummy_transducer(hidden_size=3)
    length_norm = False
    return TransducerBeamDecoderDummy(
        blank_index=blank_index,
        model=model,
        beam_width=2,
        length_norm=length_norm,
        max_symbols_per_step=max_symbols_per_step,
    )


# Tests -----------------------------------------------------------------------


def test_beam_search_single_step(decoder=get_fixed_decoder()):
    indata = torch.tensor([[[0.3, 0.6, 0.1]]])  # (1, 1, 3)
    indata = indata.unsqueeze(3)  # B, C, F, T = (1, 1, 3, 1)
    lengths = torch.IntTensor([1])
    inp = (indata, lengths)
    assert decoder.decode(inp) == [1, 1]


def test_beam_search_multi_step(decoder=get_fixed_decoder()):
    indata = torch.tensor(
        [[[[0.3, 0.6, 0.1], [0.3, 0.6, 0.1]]]]
    )  # (1, 1, 2, 3)
    indata = indata.transpose(2, 3)
    assert indata.shape == (1, 1, 3, 2)  # B, C, F, T = (1, 1, 3, 2)
    lengths = torch.IntTensor([2])
    assert decoder.decode((indata, lengths)) == [1, 1, 1, 1, 1]


def test_beam_search_limit_symbols_per_step(
    decoder=get_fixed_decoder(max_symbols_per_step=1),
):
    indata = torch.tensor(
        [[[[0.3, 0.6, 0.1], [0.3, 0.6, 0.1]]]]
    )  # (1, 1, 2, 3)
    indata = indata.transpose(2, 3)
    assert indata.shape == (1, 1, 3, 2)  # B, C, F, T = (1, 1, 3, 2)
    lengths = torch.IntTensor([2])
    assert decoder.decode((indata, lengths)) == [1, 1]


def test_multi_element_batch(decoder=get_fixed_decoder()):

    indata = torch.tensor(
        [
            [[[0.3, 0.6, 0.1], [0.3, 0.6, 0.1]]],
            [[[0.3, 0.6, 0.1], [0.3, 0.6, 0.1]]],
        ]
    )  # (2, 1, 2, 3)
    indata = indata.transpose(2, 3)

    assert indata.shape == (2, 1, 3, 2)  # B, C, F, T = (2, 1, 3, 2)
    lengths = torch.IntTensor([2, 1])
    assert decoder((indata, lengths)) == [[1, 1, 1, 1, 1], [1, 1]]


def test_preserves_training_state(decoder=get_fixed_decoder()):
    indata = torch.tensor([[[0.3, 0.6, 0.1]]])  # (1, 1, 3)
    indata = indata.unsqueeze(3)  # B, C, F, T = (1, 1, 3, 1)
    lengths = torch.IntTensor([1])

    decoder._model.train()
    decoder((indata, lengths))
    assert decoder._model.training

    decoder._model.eval()
    decoder((indata, lengths))
    assert not decoder._model.training
