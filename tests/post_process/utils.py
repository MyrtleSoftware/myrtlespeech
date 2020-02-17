from typing import Optional
from typing import Tuple

import torch
from myrtlespeech.model.rnn import RNNState
from myrtlespeech.model.rnn_t import RNNTEncoder
from myrtlespeech.model.rnn_t import RNNTJointNet
from myrtlespeech.model.rnn_t import RNNTPredictNet
from myrtlespeech.model.transducer import Transducer


def get_dummy_transducer(hidden_size=3):
    """Produces a Transducer for post_process testing."""
    encoder = DummyTransducerEncoder()
    predict_net = DummyPredictNet(pred_nn=PredNN(hidden_size=hidden_size))
    joint_net = DummyJointNet()
    model = Transducer(
        encoder=encoder, predict_net=predict_net, joint_net=joint_net
    )
    model.eval()
    return model


class DummyTransducerEncoder(RNNTEncoder):
    r"""An :py:class:`RNNTEncoder` with overridden forward method.

    This is necessary because in this worked example the audio inputs are
    simply returned as-is (with the dimension change required by the
    :py:class:`Transducer` API).
    """

    def __init__(self):
        rnn1 = torch.nn.RNN(2, 2)  # just usef for initialize
        super().__init__(rnn1=rnn1)

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        hx: Optional[RNNState] = None,
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

        if hx is None:
            hx = torch.zeros(1, 1, 1)

        return (h, x[1]), hx


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


class DummyPredictNet(RNNTPredictNet):
    """An :py:class:`RNNTPredictNet` with overridden forward method.

    Note that ``embedding=None`` will be passed to :py:meth:`super.__init__`
    as the :py:meth:`embed` method that calls forward on
    :py:class:`embedding` in :py:class:`RNNTPredictNet` is overridden by
    ``dummy_embed`` below.

    Args:
        pred_nn: module with same :py:meth:`__call__` API as `pred_nn` in
            :py:class:`RNNTPredictNet`.
    """

    def __init__(self, pred_nn):
        super().__init__(embedding=None, pred_nn=pred_nn)


def dummy_embed(x):
    """Overridden :py:meth:`embed`."""
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

    def __call__(self, x, hx=None):
        """Replicates `pred_nn` :py:meth:`forward` method.

        This uses the hidden state to decide which characters to
        upweight at a given timestep.

        Args:
            x: See :py:class:`RNN` with `batch_first=True`.

        Returns:
            x: See :py:class:`RNN` with `batch_first=True`.
        """
        embedded, lengths = x

        B, U_, H = embedded.shape

        assert B == U_ and B == 1, "Currently only supports batch=seq len== 1"

        if hx is None:
            # `hx` is the index of char to upweight.
            # In first instance upweight character at index 1
            hx = torch.IntTensor([1])
            hx = hx.view((1, 1, 1))
        if embedded.squeeze().int()[0] == 0:
            # i.e. if this is the SOS,
            # assign probability of 0.2 to all three chars:
            res = torch.ones(3, dtype=torch.float32) * 0.2
        else:
            res = torch.ones(3, dtype=torch.float32) * 0.1
            state_to_upweight = embedded.squeeze().int()[1]  # 0 or 1
            res[state_to_upweight] += 0.3

        res[hx.item()] += 0.4
        out = torch.log(res)
        # blow up to full dimension
        out = out.unsqueeze(0).unsqueeze(0)

        hid = (hx + 1) % 3

        return (out, lengths), hid
