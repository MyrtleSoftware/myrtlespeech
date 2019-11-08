from typing import Optional
from typing import Tuple

import torch
from myrtlespeech.builders.fully_connected import build as build_fully_connected
from myrtlespeech.builders.rnn import build as build_rnn
from myrtlespeech.data.stack import StackTime
from myrtlespeech.model.rnn_t import RNNT
from myrtlespeech.model.rnn_t import RNNTEncoder
from myrtlespeech.model.utils import Lambda
from myrtlespeech.protos import rnn_t_encoder_pb2
from myrtlespeech.protos import rnn_t_pb2
from torch import nn


def build(
    rnn_t_cfg: rnn_t_pb2.RNNT,
    input_features: int,
    input_channels: int,
    vocab_size: int,
) -> RNNT:
    """Returns a :py:class:`.RNNT` based on the config.

    Args:
        rnn_t_cfg: An ``RNNT`` protobuf object containing
            the config for the desired :py:class:`torch.nn.Module`.

        input_features: The number of features for the input.

        input_channels: The number of channels in the input.

        output_features: The number of output features.

    Returns:
        A :py:class:`.RNNT` based on the config.

    Example:
        >>> from google.protobuf import text_format
        >>> cfg_text = '''
        ... rnn_t_encoder {
        ...    fc1 {
        ...       num_hidden_layers: 1;
        ...       hidden_size: 1152;
        ...       activation {
        ...         hardtanh {
        ...           min_val: 0.0;
        ...           max_val: 20.0;
        ...           }
        ...         }
        ...       }
        ...     rnn1 {
        ...         rnn_type: LSTM;
        ...         hidden_size: 1152;
        ...         num_layers: 2;
        ...         bias: true;
        ...         bidirectional: false;
        ...             }
        ...
        ...      fc2 {
        ...        num_hidden_layers: 1;
        ...         hidden_size: 1152;
        ...         activation {
        ...           hardtanh {
        ...             min_val: 0.0;
        ...             max_val: 20.0;
        ...             }
        ...           }
        ...         }
        ...     }
        ... dec_rnn {
        ...     rnn_type: LSTM;
        ...     hidden_size: 256;
        ...     num_layers: 2;
        ...     bias: true;
        ...     bidirectional: false;
        ...     batch_first: true;
        ...    }
        ...
        ... fully_connected {
        ...     num_hidden_layers: 1;
        ...     hidden_size: 512;
        ...     activation {
        ...       hardtanh {
        ...         min_val: 0.0;
        ...         max_val: 20.0;
        ...         }
        ...       }
        ...     }
        ... '''
        >>> cfg = text_format.Merge(
        ...             cfg_text,
        ...             rnn_t_pb2.RNNT()
        ... )
        >>> build(cfg, input_features=80, input_channels=5, vocab_size=28)
        RNNT(
          (encode): RNNTEncoder(
            (fc1): FullyConnected(
              (fully_connected): Sequential(
                (0): Linear(in_features=400, out_features=1152, bias=True)
                (1): Hardtanh(min_val=0.0, max_val=20.0)
                (2): Linear(in_features=1152, out_features=1152, bias=True)
              )
            )
            (rnn1): RNN(
              (rnn): LSTM(1152, 1152, num_layers=2)
            )
            (fc2): FullyConnected(
              (fully_connected): Sequential(
                (0): Linear(in_features=1152, out_features=1152, bias=True)
                (1): Hardtanh(min_val=0.0, max_val=20.0)
                (2): Linear(in_features=1152, out_features=576, bias=True)
              )
            )
          )
          (predict_net): ModuleDict(
            (dec_rnn): RNN(
              (rnn): LSTM(256, 256, num_layers=2, batch_first=True)
            )
            (embed): Embedding(28, 256)
          )
          (joint_net): ModuleDict(
            (fully_connected): FullyConnected(
              (fully_connected): Sequential(
                (0): Linear(in_features=832, out_features=512, bias=True)
                (1): Hardtanh(min_val=0.0, max_val=20.0)
                (2): Linear(in_features=512, out_features=29, bias=True)
              )
            )
          )
        )

    """

    encoder, encoder_out = build_rnnt_enc(
        rnn_t_cfg.rnn_t_encoder, input_features * input_channels
    )

    ##decoder/prediction network
    # can get embedding dims from the rnnt
    embedding = nn.Embedding(vocab_size, rnn_t_cfg.dec_rnn.hidden_size)
    dec_rnn, prediction_out = build_rnn(
        rnn_t_cfg.dec_rnn, rnn_t_cfg.dec_rnn.hidden_size
    )

    ##joint
    fc_in_dim = encoder_out + prediction_out  # features are concatenated

    fully_connected = build_fully_connected(
        rnn_t_cfg.fully_connected,
        input_features=fc_in_dim,
        output_features=vocab_size + 1,
    )
    rnnt = RNNT(encoder, embedding, dec_rnn, fully_connected)
    if torch.cuda.is_available():
        rnnt.cuda()
    return rnnt


def build_rnnt_enc(
    rnn_t_enc: rnn_t_encoder_pb2.RNNTEncoder, input_features: int
) -> Tuple[RNNTEncoder, int]:
    """Returns a :py:class:`.RNNTEncoder` based on the config.

    Args:
        rnn_t_cfg: An ``RNNT`` protobuf object containing
            the config for the desired :py:class:`torch.nn.Module`.

        input_features: The number of features for the input.

    Returns:
        A Tuple where the first element is an :py:class:`.RNNTEncoder` based
            on the config and the second element is the encoder output feature
            size. See :py:class:`.RNNTEncoder` docstrings for more information.

    Example:
        >>> from google.protobuf import text_format
        >>> cfg_text = '''
        ... fc1 {
        ... num_hidden_layers: 1;
        ... hidden_size: 1152;
        ... activation {
        ...   hardtanh {
        ...     min_val: 0.0;
        ...     max_val: 20.0;
        ...     }
        ...   }
        ... }
        ... rnn1 {
        ...   rnn_type: LSTM;
        ...   hidden_size: 1152;
        ...   num_layers: 2;
        ...   bias: true;
        ...   bidirectional: false;
        ...       }
        ...
        ... fc2 {
        ... num_hidden_layers: 1;
        ... hidden_size: 1152;
        ... activation {
        ...   hardtanh {
        ...     min_val: 0.0;
        ...     max_val: 20.0;
        ...     }
        ...   }
        ... }
        ... '''
        >>> cfg = text_format.Merge(
        ...             cfg_text,
        ...             rnn_t_encoder_pb2.RNNTEncoder()
        ... )
        >>> encoder, out_features = build_rnnt_enc(cfg, input_features=400)
        >>> encoder
        RNNTEncoder(
          (fc1): FullyConnected(
            (fully_connected): Sequential(
              (0): Linear(in_features=400, out_features=1152, bias=True)
              (1): Hardtanh(min_val=0.0, max_val=20.0)
              (2): Linear(in_features=1152, out_features=1152, bias=True)
            )
          )
          (rnn1): RNN(
            (rnn): LSTM(1152, 1152, num_layers=2)
          )
          (fc2): FullyConnected(
            (fully_connected): Sequential(
              (0): Linear(in_features=1152, out_features=1152, bias=True)
              (1): Hardtanh(min_val=0.0, max_val=20.0)
              (2): Linear(in_features=1152, out_features=576, bias=True)
            )
          )
        )
        >>> out_features
        576
    """

    # maybe add fc1:
    fc1: Optional[torch.nn.Module] = None

    if rnn_t_enc.HasField("fc1"):
        output_features = rnn_t_enc.rnn1.hidden_size
        fc1 = build_fully_connected(
            rnn_t_enc.fc1,
            input_features=input_features,
            output_features=output_features,
        )
        input_features = output_features

    rnn1, rnn1_out_features = build_rnn(rnn_t_enc.rnn1, input_features)

    if rnn_t_enc.time_reduction_factor == 0:  # default value (i.e. not set)
        assert rnn_t_enc.HasField("rnn2") is False

        time_reducer, rnn2 = None, None
        reduction = 1
        rnn_out_features = rnn1_out_features

    else:
        time_reduction_factor = rnn_t_enc.time_reduction_factor

        assert (
            time_reduction_factor > 1
        ), "time_reduction_factor must be an integer > 1 but is = {time_reduction_factor}"

        reduction = rnn_t_enc.time_reduction_factor

        time_reducer = Lambda(StackTime(reduction))

        rnnt_input_features = rnn1_out_features * reduction

        rnn2, rnn_out_features = build_rnn(rnn_t_enc.rnn2, rnnt_input_features)

    # maybe add fc2:
    fc2: Optional[torch.nn.Module] = None
    if rnn_t_enc.HasField("fc2"):
        # This layer halves feature size if possible
        out_features = rnn_out_features // 2
        out_features = out_features if out_features > 0 else 1

        fc2 = build_fully_connected(
            rnn_t_enc.fc2,
            input_features=rnn_out_features,
            output_features=out_features,
        )
    else:
        out_features = rnn_out_features

    encoder = RNNTEncoder(
        rnn1=rnn1,
        fc1=fc1,
        time_reducer=time_reducer,
        time_reduction_factor=reduction,
        rnn2=rnn2,
        fc2=fc2,
    )

    return encoder, out_features