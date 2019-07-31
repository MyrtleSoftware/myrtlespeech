from typing import Optional

import pytest
import torch
import hypothesis.strategies as st
from hypothesis import assume, given

from myrtlespeech.builders.encoder import build
from myrtlespeech.model.encoder.cnn_rnn_encoder import CNNRNNEncoder
from myrtlespeech.model.encoder.encoder import Encoder
from myrtlespeech.model.encoder.vgg import vgg_output_size
from myrtlespeech.protos import encoder_pb2
from tests.builders.test_rnn import rnn_match_cfg
from tests.builders.test_vgg import vgg_match_cfg
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


def cnn_rnn_encoder_match_cfg(
    encoder: CNNRNNEncoder,
    encoder_cfg: encoder_pb2.Encoder.CNNRNNEncoder,
    input_features: int,
    input_channels: int,
    seq_len_support: bool,
) -> None:
    # check cnn config
    if encoder_cfg.HasField("no_cnn"):
        assert encoder.cnn is None
    elif encoder_cfg.HasField("vgg"):
        vgg_match_cfg(  # type: ignore
            encoder.cnn, encoder_cfg.vgg, input_channels, seq_len_support
        )
        out_size = vgg_output_size(
            encoder.cnn, torch.Size([-1, input_channels, input_features, -1])
        )
        input_features = out_size[1] * out_size[2]
    else:
        raise ValueError("expected either no_cnn or vgg to be set")

    # check rnn config
    if encoder_cfg.HasField("no_rnn"):
        assert encoder.rnn is None
    elif encoder_cfg.HasField("rnn"):
        rnn_match_cfg(encoder.rnn, encoder_cfg.rnn, input_features)
    else:
        raise ValueError("expected either no_rnn or rnn to be set")


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
def test_unknown_cnn_raises_value_error(
    encoder_cfg: encoder_pb2.Encoder,
    input_features: int,
    input_channels: int,
    seq_len_support: bool,
) -> None:
    """Ensures ValueError is raised when cnn_rnn_encoder's cnn is not supported.

    This can occur when the protobuf is updated and build is not.
    """
    assume(encoder_cfg.WhichOneof("supported_encoders") == "cnn_rnn_encoder")
    encoder_cfg.cnn_rnn_encoder.ClearField(
        encoder_cfg.cnn_rnn_encoder.WhichOneof("supported_cnns")
    )
    with pytest.raises(ValueError):
        build(encoder_cfg, input_features, input_channels, seq_len_support)


@given(
    encoder_cfg=encoders(),
    input_features=st.integers(min_value=1, max_value=32),
    input_channels=st.integers(min_value=1, max_value=8),
    seq_len_support=st.booleans(),
)
def test_unknown_rnn_raises_value_error(
    encoder_cfg: encoder_pb2.Encoder,
    input_features: int,
    input_channels: int,
    seq_len_support: bool,
) -> None:
    """Ensures ValueError is raised when cnn_rnn_encoder's rnn is not supported.

    This can occur when the protobuf is updated and build is not.
    """
    assume(encoder_cfg.WhichOneof("supported_encoders") == "cnn_rnn_encoder")
    encoder_cfg.cnn_rnn_encoder.ClearField(
        encoder_cfg.cnn_rnn_encoder.WhichOneof("supported_rnns")
    )
    with pytest.raises(ValueError):
        build(encoder_cfg, input_features, input_channels, seq_len_support)
