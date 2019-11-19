import math

import torch
from myrtlespeech.builders.activation import build as build_activation
from myrtlespeech.builders.fully_connected import (
    build as build_fully_connected,
)
from myrtlespeech.builders.lookahead import build as build_lookahead
from myrtlespeech.builders.rnn import build as build_rnn
from myrtlespeech.model.cnn import Conv1dTo2d
from myrtlespeech.model.cnn import Conv2dTo1d
from myrtlespeech.model.cnn import MaskConv1d
from myrtlespeech.model.cnn import MaskConv2d
from myrtlespeech.model.cnn import out_lens
from myrtlespeech.model.cnn import PaddingMode
from myrtlespeech.model.deep_speech_2 import DeepSpeech2
from myrtlespeech.model.seq_len_wrapper import SeqLenWrapper
from myrtlespeech.protos import conv_layer_pb2
from myrtlespeech.protos import deep_speech_2_pb2


def build(
    deep_speech_2_cfg: deep_speech_2_pb2.DeepSpeech2,
    input_features: int,
    input_channels: int,
    output_features: int,
) -> DeepSpeech2:
    """Returns a :py:class:`.DeepSpeech2` based on the config.

    Args:
        deep_speech_2_cfg: A ``DeepSpeech2`` protobuf object containing the
            config for the desired :py:class:`.DeepSpeech2`.

        input_features: The number of input features.

        input_channels: The number of input channels.

        output_features: The number of output features.

    Returns:
        A :py:class:`.DeepSpeech2` based on the config.

    Example:
        >>> # noqa: E501
        >>> from google.protobuf import text_format
        >>> cfg_text = '''
        ... conv_block {
        ...   conv1d {
        ...     output_channels: 8;
        ...     kernel_time: 5;
        ...     stride_time: 1;
        ...     padding_mode: NONE;
        ...     bias: true;
        ...   }
        ...   activation {
        ...     relu { }
        ...   }
        ... }
        ...
        ... rnn {
        ...   rnn_type: LSTM;
        ...   hidden_size: 32;
        ...   num_layers: 1;
        ...   bias: true;
        ...   bidirectional: true;
        ... }
        ...
        ... lookahead_block {
        ...   no_lookahead { }
        ...   activation {
        ...     identity { }
        ...   }
        ... }
        ...
        ... fully_connected {
        ...   num_hidden_layers: 0;
        ...   activation {
        ...     identity { }
        ...   }
        ... }
        ... '''
        >>> cfg = text_format.Merge(
        ...     cfg_text,
        ...     deep_speech_2_pb2.DeepSpeech2()
        ... )
        >>> build(cfg, input_features=3, input_channels=1, output_features=4)
        DeepSpeech2(
          (cnn): Sequential(
            (0): Conv2dTo1d(seq_len_support=True)
            (1): MaskConv1d(3, 8, kernel_size=(5,), stride=(1,), padding_mode=PaddingMode.NONE)
            (2): SeqLenWrapper(
              (module): ReLU()
              (seq_lens_fn): Identity()
            )
            (3): Conv1dTo2d(seq_len_support=True)
          )
          (rnn): Sequential(
            (0): RNN(
              (rnn): LSTM(8, 32, bidirectional=True)
            )
          )
          (fully_connected): Sequential(
            (0): FullyConnected(
              (fully_connected): Linear(in_features=64, out_features=4, bias=True)
            )
          )
        )
    """
    cnn, cnn_out_features = _build_cnn(
        deep_speech_2_cfg.conv_block, input_features, input_channels
    )

    rnn, rnn_out_features = build_rnn(
        deep_speech_2_cfg.rnn, input_features=cnn_out_features
    )

    if deep_speech_2_cfg.lookahead_block.HasField("lookahead"):
        lookahead = build_lookahead(
            deep_speech_2_cfg.lookahead_block.lookahead,
            input_features=rnn_out_features,
        )
        activation = SeqLenWrapper(
            build_activation(deep_speech_2_cfg.lookahead_block.activation),
            torch.nn.Identity(),
        )
        if activation != torch.nn.Identity:
            lookahead = torch.nn.Sequential(lookahead, activation)
    else:
        lookahead = None  # type: ignore

    fully_connected = build_fully_connected(
        deep_speech_2_cfg.fully_connected,
        input_features=rnn_out_features,
        output_features=output_features,
    )

    return DeepSpeech2(cnn, rnn, lookahead, fully_connected)


def _build_cnn(conv_blocks, input_features: int, input_channels: int):
    act_dims = 4  # batch, channels, features, seq_len

    layers = []
    for conv_block in conv_blocks:
        convnd_str = conv_block.WhichOneof("convnd")

        if convnd_str == "conv1d":
            if act_dims == 4:
                layers.append(Conv2dTo1d())
                act_dims = 3
                input_channels *= input_features
                input_features = 1

            conv_cfg = conv_block.conv1d

            if conv_cfg.padding_mode == conv_layer_pb2.PADDING_MODE.NONE:
                padding_mode = PaddingMode.NONE
            elif conv_cfg.padding_mode == conv_layer_pb2.PADDING_MODE.SAME:
                padding_mode = PaddingMode.SAME

            layers.append(
                MaskConv1d(
                    in_channels=input_channels,
                    out_channels=conv_cfg.output_channels,
                    kernel_size=conv_cfg.kernel_time,
                    stride=conv_cfg.stride_time,
                    padding_mode=padding_mode,
                    bias=conv_cfg.bias,
                )
            )

            input_channels = conv_cfg.output_channels

        elif convnd_str == "conv2d":
            if act_dims == 3:
                layers.append(Conv1dTo2d())
                act_dims = 4
                input_features = input_channels
                input_channels = 1

            conv_cfg = conv_block.conv2d

            if conv_cfg.padding_mode == conv_layer_pb2.PADDING_MODE.NONE:
                padding_mode = PaddingMode.NONE
                input_features = out_lens(
                    torch.tensor([input_features]),
                    kernel_size=conv_cfg.kernel_feature,
                    stride=conv_cfg.stride_feature,
                    dilation=1,
                    padding=0,
                ).item()
            elif conv_cfg.padding_mode == conv_layer_pb2.PADDING_MODE.SAME:
                padding_mode = PaddingMode.SAME
                input_features = math.ceil(
                    input_features / conv_cfg.stride_feature
                )

            layers.append(
                MaskConv2d(
                    in_channels=input_channels,
                    out_channels=conv_cfg.output_channels,
                    kernel_size=[
                        conv_cfg.kernel_feature,
                        conv_cfg.kernel_time,
                    ],
                    stride=[conv_cfg.stride_feature, conv_cfg.stride_time],
                    padding_mode=padding_mode,
                    bias=conv_cfg.bias,
                )
            )

            input_channels = conv_cfg.output_channels

        layers.append(
            SeqLenWrapper(
                build_activation(conv_block.activation), torch.nn.Identity()
            )
        )

    if act_dims == 3:
        layers.append(Conv1dTo2d())
        input_features = input_channels
        input_channels = 1

    return torch.nn.Sequential(*layers), input_features * input_channels
