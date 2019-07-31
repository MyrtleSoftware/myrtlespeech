import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given
from myrtlespeech.builders.cnn_rnn_encoder import build
from myrtlespeech.model.encoder_decoder.encoder.cnn_rnn_encoder import (
    CNNRNNEncoder,
)
from myrtlespeech.model.encoder_decoder.encoder.vgg import vgg_output_size
from myrtlespeech.protos import cnn_rnn_encoder_pb2

from tests.builders.test_rnn import rnn_match_cfg
from tests.builders.test_vgg import vgg_match_cfg
from tests.protos.test_cnn_rnn_encoder import cnn_rnn_encoders


# Utilities -------------------------------------------------------------------


def cnn_rnn_encoder_match_cfg(
    cnn_rnn_encoder: CNNRNNEncoder,
    cnn_rnn_encoder_cfg: cnn_rnn_encoder_pb2.CNNRNNEncoder,
    input_features: int,
    input_channels: int,
    seq_len_support: bool,
) -> None:
    # check cnn config
    if cnn_rnn_encoder_cfg.HasField("no_cnn"):
        assert cnn_rnn_encoder.cnn is None
    elif cnn_rnn_encoder_cfg.HasField("vgg"):
        vgg_match_cfg(  # type: ignore
            cnn_rnn_encoder.cnn,
            cnn_rnn_encoder_cfg.vgg,
            input_channels,
            seq_len_support,
        )
        out_size = vgg_output_size(
            cnn_rnn_encoder.cnn,
            torch.Size([-1, input_channels, input_features, -1]),
        )
        input_features = out_size[1] * out_size[2]
    else:
        raise ValueError("expected either no_cnn or vgg to be set")

    # check rnn config
    if cnn_rnn_encoder_cfg.HasField("no_rnn"):
        assert cnn_rnn_encoder.rnn is None
    elif cnn_rnn_encoder_cfg.HasField("rnn"):
        rnn_match_cfg(
            cnn_rnn_encoder.rnn, cnn_rnn_encoder_cfg.rnn, input_features
        )
    else:
        raise ValueError("expected either no_rnn or rnn to be set")


# Tests -----------------------------------------------------------------------


@given(
    cnn_rnn_encoder_cfg=cnn_rnn_encoders(),
    input_features=st.integers(min_value=1, max_value=32),
    input_channels=st.integers(min_value=1, max_value=8),
    seq_len_support=st.booleans(),
)
def test_build_cnn_rnn_encoder_returns_correct_structure_and_out_features(
    cnn_rnn_encoder_cfg: cnn_rnn_encoder_pb2.CNNRNNEncoder,
    input_features: int,
    input_channels: int,
    seq_len_support: bool,
) -> None:
    """Ensures tuple returned by ``build`` has correct structure."""
    cnn_rnn_encoder, _ = build(
        cnn_rnn_encoder_cfg, input_features, input_channels, seq_len_support
    )
    cnn_rnn_encoder_match_cfg(
        cnn_rnn_encoder,
        cnn_rnn_encoder_cfg,
        input_features,
        input_channels,
        seq_len_support,
    )


@given(
    cnn_rnn_encoder_cfg=cnn_rnn_encoders(),
    input_features=st.integers(min_value=1, max_value=32),
    input_channels=st.integers(min_value=1, max_value=8),
    seq_len_support=st.booleans(),
)
def test_unknown_cnn_raises_value_error(
    cnn_rnn_encoder_cfg: cnn_rnn_encoder_pb2.CNNRNNEncoder,
    input_features: int,
    input_channels: int,
    seq_len_support: bool,
) -> None:
    """Ensures ValueError is raised when cnn_rnn_encoder's cnn is not supported.

    This can occur when the protobuf is updated and build is not.
    """
    cnn_rnn_encoder_cfg.ClearField(
        cnn_rnn_encoder_cfg.WhichOneof("supported_cnns")
    )
    with pytest.raises(ValueError):
        build(
            cnn_rnn_encoder_cfg, input_features, input_channels, seq_len_support
        )


@given(
    cnn_rnn_encoder_cfg=cnn_rnn_encoders(),
    input_features=st.integers(min_value=1, max_value=32),
    input_channels=st.integers(min_value=1, max_value=8),
    seq_len_support=st.booleans(),
)
def test_unknown_rnn_raises_value_error(
    cnn_rnn_encoder_cfg: cnn_rnn_encoder_pb2.CNNRNNEncoder,
    input_features: int,
    input_channels: int,
    seq_len_support: bool,
) -> None:
    """Ensures ValueError is raised when cnn_rnn_encoder's rnn is not supported.

    This can occur when the protobuf is updated and build is not.
    """
    cnn_rnn_encoder_cfg.ClearField(
        cnn_rnn_encoder_cfg.WhichOneof("supported_rnns")
    )
    with pytest.raises(ValueError):
        build(
            cnn_rnn_encoder_cfg, input_features, input_channels, seq_len_support
        )
