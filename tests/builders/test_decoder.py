import torch
import hypothesis.strategies as st
from hypothesis import assume, given

from myrtlespeech.builders.decoder import build as build_decoder
from myrtlespeech.protos import decoder_pb2
from myrtlespeech.protos import fully_connected_pb2
from tests.protos.test_decoder import decoders
from tests.builders.test_fully_connected import fully_connected_module_match_cfg


# Utilities -------------------------------------------------------------------


def decoder_module_match_cfg(
    decoder: torch.nn.Module,
    decoder_cfg: decoder_pb2.Decoder,
    input_features: int,
    output_features: int,
) -> None:
    """Ensures Decoder matches protobuf configuration."""
    assert isinstance(decoder, torch.nn.Module)

    if decoder_cfg.HasField("fully_connected"):
        fully_connected_module_match_cfg(  # type: ignore
            decoder,
            decoder_cfg.fully_connected,
            input_features,
            output_features,
        )
    else:
        raise ValueError("expected fully_connected to be set")


# Tests -----------------------------------------------------------------------


@given(
    decoder_cfg=decoders(),
    input_features=st.integers(min_value=1, max_value=32),
    output_features=st.integers(min_value=1, max_value=32),
)
def test_build_decoder_returns_correct_module_structure(
    decoder_cfg: decoder_pb2.Decoder, input_features: int, output_features: int
) -> None:
    """Ensures Module returned by ``build`` has correct structure."""
    if decoder_cfg.HasField("fully_connected"):
        fc = decoder_cfg.fully_connected
        if fc.num_hidden_layers == 0:
            assume(fc.hidden_size is None)
            assume(
                fc.hidden_activation_fn
                == fully_connected_pb2.FullyConnected.NONE
            )
    decoder = build_decoder(decoder_cfg, input_features, output_features)
    decoder_module_match_cfg(
        decoder, decoder_cfg, input_features, output_features
    )
