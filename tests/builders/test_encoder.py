import hypothesis.strategies as st
import pytest
from hypothesis import given
from myrtlespeech.builders.encoder import build
from myrtlespeech.model.encoder_decoder.encoder.cnn_rnn_encoder import (
    CNNRNNEncoder,
)
from myrtlespeech.model.encoder_decoder.encoder.encoder import Encoder
from myrtlespeech.protos import encoder_pb2

from tests.builders.test_cnn_rnn_encoder import cnn_rnn_encoder_match_cfg
from tests.protos.test_encoder import encoders


# Utilities -------------------------------------------------------------------


def encoder_match_cfg(
    encoder: Encoder,
    encoder_cfg: encoder_pb2.Encoder,
    input_features: int,
    input_channels: int,
    seq_len_support: bool,
) -> None:
    """Ensures Encoder matches protobuf configuration."""
    assert isinstance(encoder, Encoder)

    if isinstance(encoder, CNNRNNEncoder):
        cnn_rnn_encoder_match_cfg(
            encoder,
            encoder_cfg.cnn_rnn_encoder,
            input_features,
            input_channels,
            seq_len_support,
        )
    else:
        raise ValueError(f"unsupported encoder {type(encoder)}")


# Tests -----------------------------------------------------------------------


@given(
    encoder_cfg=encoders(),
    input_features=st.integers(min_value=1, max_value=32),
    input_channels=st.integers(min_value=1, max_value=8),
    seq_len_support=st.booleans(),
)
def test_build_encoder_returns_correct_module_structure_and_out_features(
    encoder_cfg: encoder_pb2.Encoder,
    input_features: int,
    input_channels: int,
    seq_len_support: bool,
) -> None:
    """Ensures tuple returned by ``build`` has correct structure."""
    encoder, _ = build(
        encoder_cfg, input_features, input_channels, seq_len_support
    )
    encoder_match_cfg(
        encoder, encoder_cfg, input_features, input_channels, seq_len_support
    )


@given(
    encoder_cfg=encoders(),
    input_features=st.integers(min_value=1, max_value=32),
    input_channels=st.integers(min_value=1, max_value=8),
    seq_len_support=st.booleans(),
)
def test_unknown_supported_encoder_raises_value_error(
    encoder_cfg: encoder_pb2.Encoder,
    input_features: int,
    input_channels: int,
    seq_len_support: bool,
) -> None:
    """Ensures ValueError is raised when encoder's supported_encoders invalid.

    This can occur when the protobuf is updated and build is not.
    """
    encoder_cfg.ClearField(encoder_cfg.WhichOneof("supported_encoders"))
    with pytest.raises(ValueError):
        build(encoder_cfg, input_features, input_channels, seq_len_support)
