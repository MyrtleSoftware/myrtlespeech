from typing import Tuple

import torch

from myrtlespeech.builders.rnn import build as build_rnn
from myrtlespeech.builders.vgg import build as build_vgg
from myrtlespeech.model.encoder_decoder.encoder.cnn_rnn_encoder import (
    CNNRNNEncoder,
)
from myrtlespeech.model.encoder_decoder.encoder.vgg import vgg_output_size
from myrtlespeech.protos import cnn_rnn_encoder_pb2


def build(
    cnn_rnn_encoder_cfg: cnn_rnn_encoder_pb2.CNNRNNEncoder,
    input_features: int,
    input_channels: int = 1,
    seq_len_support: bool = False,
) -> Tuple[CNNRNNEncoder, int]:
    """Returns an :py:class:`.CNNRNNEncoder` based on the given config.

    Args:
        cnn_rnn_encoder_cfg: A
            :py:class:`myrtlespeech.protos.encoder_pb2.Encoder.CNNRNNEncoder`
            protobuf object containing the config for the desired
            :py:class:`.CNNRNNEncoder`.

        input_features: The number of features for the input.

        input_channels: The number of channels for the input.

    Returns:
        A tuple containing an :py:class:`.CNNRNNEncoder` based on the config
        and the number of output features.

    Example:
        >>> from google.protobuf import text_format
        >>> encoder_cfg_text = '''
        ... vgg {
        ...   vgg_config: A;
        ...   batch_norm: false;
        ...   use_output_from_block: 2;
        ... }
        ... rnn {
        ...   rnn_type: LSTM;
        ...   hidden_size: 1024;
        ...   num_layers: 5;
        ...   bias: true;
        ...   bidirectional: true;
        ... }
        ... '''
        >>> encoder_cfg = text_format.Merge(
        ...     encoder_cfg_text,
        ...     cnn_rnn_encoder_pb2.CNNRNNEncoder()
        ... )
        >>> build(encoder_cfg, input_features=10, input_channels=3)
        (CNNRNNEncoder(
          (cnn): Sequential(
            (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(inplace)
            (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (4): ReLU(inplace)
            (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (7): ReLU(inplace)
            (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (9): ReLU(inplace)
            (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          )
          (cnn_to_rnn): Lambda()
          (rnn): RNN(
            (rnn): LSTM(256, 1024, num_layers=5, bidirectional=True)
          )
        ), 2048)
    """
    # build cnn, if any
    cnn_choice = cnn_rnn_encoder_cfg.WhichOneof("supported_cnns")
    if cnn_choice == "no_cnn":
        cnn = None
    elif cnn_choice == "vgg":
        cnn = build_vgg(  # type: ignore
            cnn_rnn_encoder_cfg.vgg,
            input_channels,
            seq_len_support=seq_len_support,
        )
        # update number of features based on vgg output
        out_size = vgg_output_size(
            cnn if not seq_len_support else cnn.module,
            torch.Size([-1, input_channels, input_features, -1]),
        )
        input_features = out_size[1] * out_size[2]
    else:
        raise ValueError(f"{cnn_choice} not supported")

    # build rnn, if any
    rnn_choice = cnn_rnn_encoder_cfg.WhichOneof("supported_rnns")
    if rnn_choice == "no_rnn":
        rnn = None
        output_features = input_features
    elif rnn_choice == "rnn":
        rnn = build_rnn(cnn_rnn_encoder_cfg.rnn, input_features)
        output_features = rnn.rnn.hidden_size
        if rnn.rnn.bidirectional:
            output_features *= 2
    else:
        raise ValueError(f"{rnn_choice} not supported")

    return CNNRNNEncoder(cnn=cnn, rnn=rnn), output_features
