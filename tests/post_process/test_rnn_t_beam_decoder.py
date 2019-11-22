from typing import Tuple

import torch
from myrtlespeech.model.rnn_t import RNNT
from myrtlespeech.post_process.rnn_t_beam_decoder import RNNTBeamDecoder


# Fixtures and Strategies -----------------------------------------------------
class RNN:
    def __init__(self, hidden_size=3):
        self.hidden_size = hidden_size


class DecRNN:
    def __init__(self, hidden_size=3):
        self.rnn = RNN(hidden_size)

    def __call__(self, x):
        """TODO: The way this works is by using the hidden state to decide which
        chars to upweight at a given timestep.
         ``[B, U + 1, H]``"""

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

        assert B == U_ and B == 1, "Currently only supports batch=seq len == 1"

        lengths = x[1]

        if state is None:
            # `state` is the index of char to upweight.
            # In first instance upweight character at index 1
            state = 1
        if embedded.squeeze().int()[0] == 0:
            # i.e. if this is the SOS,
            # assign probability of 0.2 to all three chars:
            res = torch.ones(3, dtype=torch.float32) * 0.2
        else:
            res = torch.ones(3, dtype=torch.float32) * 0.1
            state_to_upweight = embedded.squeeze().int()[1]  # 0 or 1
            res[state_to_upweight] += 0.3

        res[state] += 0.4
        out, hid = torch.log(res), (state + 1) % 3
        # blow up to full dimension
        out = out.unsqueeze(0).unsqueeze(0)

        if return_tuple:
            return (out, hid), lengths
        else:
            return out, lengths


class DummyRNNTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_cuda = False
        self.dec_rnn = DecRNN(3)

    def forward(self):
        raise NotImplementedError

    def encode(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ``[B, C, H, T]`` -> ``[T, B, H_out]`` """

        h = x[0]  # B, C, H, T
        B, C, F, T = h.shape

        assert (
            C == 1
        ), f"In DummyRNNTModel(), input channels must == 1 but C == {C}"
        h = h.squeeze(1)  # B, H, T
        h = h.permute(2, 0, 1)

        return h, x[1]

    def embedding(self, x):
        res = torch.tensor([1, x[0].item() + 1]).unsqueeze(0).unsqueeze(0)
        x = res, x[1]

        return x

    @staticmethod
    def _certify_inputs_forward(*args):
        return RNNT._certify_inputs_forward(*args)

    @staticmethod
    def _prepare_inputs_forward(*args):
        return RNNT._prepare_inputs_forward(*args)


class RNNTBeamDecoderDummy(RNNTBeamDecoder):
    """Decoder class which overrides _joint_step method"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _joint_step(self, enc, pred):
        """Overrride _joint_step() as it performs a log on outputs.

        TODO: remove this class and perform an exp on inputs instead? """

        f, f_lens = enc
        g, g_lens = pred

        T, B1, H1 = f.shape
        B2, U_, H2 = g.shape

        assert (
            B1 == B2
        ), "Batch size from prediction network and transcription must be equal"

        f = f.transpose(1, 0)  # (T, B, H1) -> (B, T, H1)
        f = f.unsqueeze(dim=2)  # (B, T, 1, H)

        g = g.unsqueeze(dim=1)  # (B, 1, U_, H)

        return (f + g).squeeze()


def get_fixed_decoder(max_symbols_per_step=100):
    # alphabet = ["_", "a", "b"]
    blank_index = 0
    model = DummyRNNTModel()
    length_norm = False
    return RNNTBeamDecoderDummy(
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
    decoder=get_fixed_decoder(max_symbols_per_step=1)
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
    # API for decoder requires label input to decoder (as these inputs are
    # required for the foward pass of the model). These are deleted
    # almost immediately but user must pass torch.Tensors for labels
    # and label lengths (otherwise checks will throw exceptions)
    B, C, F, T = indata.shape
    U = 1
    labels = torch.zeros(B, U)
    label_lens = torch.IntTensor([1, 1])

    assert decoder((indata, labels), (lengths, label_lens)) == [
        [1, 1, 1, 1, 1],
        [1, 1],
    ]


def test_preserves_training_state(decoder=get_fixed_decoder()):
    indata = torch.tensor([[[0.3, 0.6, 0.1]]])  # (1, 1, 3)
    indata = indata.unsqueeze(3)  # B, C, F, T = (1, 1, 3, 1)
    lengths = torch.IntTensor([1])

    decoder.model.train()
    decoder.decode((indata, lengths))
    assert decoder.model.training

    decoder.model.eval()
    decoder.decode((indata, lengths))
    assert not decoder.model.training
