import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given
from myrtlespeech.builders.decoder import build
from myrtlespeech.protos import decoder_pb2

from tests.builders.test_fully_connected import fully_connected_module_match_cfg
from tests.protos.test_decoder import decoders


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
    decoder_cfg=decoders(valid_only=True),
    input_features=st.integers(min_value=1, max_value=32),
    output_features=st.integers(min_value=1, max_value=32),
    seq_len_support=st.booleans(),
)
def test_build_decoder_returns_correct_module_structure(
    decoder_cfg: decoder_pb2.Decoder,
    input_features: int,
    output_features: int,
    seq_len_support: bool,
) -> None:
    """Ensures Module returned by ``build`` has correct structure."""
    decoder = build(
        decoder_cfg, input_features, output_features, seq_len_support
    )
    decoder_module_match_cfg(
        decoder, decoder_cfg, input_features, output_features
    )


@given(
    decoder_cfg=decoders(valid_only=True),
    input_features=st.integers(min_value=1, max_value=32),
    output_features=st.integers(min_value=1, max_value=32),
    seq_len_support=st.booleans(),
)
def test_unknown_decoder_raises_value_error(
    decoder_cfg: decoder_pb2.Decoder,
    input_features: int,
    output_features: int,
    seq_len_support: bool,
) -> None:
    """Ensures ValueError is raised when decoder is not supported.

    This can occur when the protobuf is updated and build is not.
    """
    decoder_cfg.ClearField(decoder_cfg.WhichOneof("supported_decoders"))
    with pytest.raises(ValueError):
        build(decoder_cfg, input_features, output_features, seq_len_support)
