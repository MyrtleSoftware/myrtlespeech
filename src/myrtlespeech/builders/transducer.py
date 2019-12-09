from typing import Optional
from typing import Tuple

import torch
from myrtlespeech.builders.fully_connected import (
    build as build_fully_connected,
)
from myrtlespeech.builders.rnn import build as build_rnn
from myrtlespeech.model.rnn_t import RNNTEncoder
from myrtlespeech.model.rnn_t import RNNTJointNet
from myrtlespeech.model.rnn_t import RNNTPredictNet
from myrtlespeech.model.transducer import Transducer
from myrtlespeech.protos import transducer_encoder_pb2
from myrtlespeech.protos import transducer_joint_net_pb2
from myrtlespeech.protos import transducer_pb2
from myrtlespeech.protos import transducer_predict_net_pb2
from torch import nn


def build(
    transducer_cfg: transducer_pb2.Transducer,
    input_features: int,
    input_channels: int,
    vocab_size: int,
) -> Transducer:
    r"""Returns a :py:class:`.Transducer` based on the config.

    .. note::

        This Transducer builder currently supports RNN-Transducers **only**
        and will initialise the classes in ``model/rnn_t.py``.

    Args:
        transducer_cfg: A ``Transducer`` protobuf object containing
            the config for the desired :py:class:`torch.nn.Module`.

        input_features: The number of features for the input.

        input_channels: The number of channels in the input.

        output_features: The number of output features.

    Returns:
        A :py:class:`.Transducer` based on the config.

    Example:
        >>> from google.protobuf import text_format
        >>> cfg_txt = '''
        ... transducer_encoder {
        ...       fc1 {
        ...       num_hidden_layers: 1;
        ...       hidden_size: 1152;
        ...       dropout: {value: 0.25}
        ...       activation {
        ...         hardtanh {
        ...           min_val: 0.0;
        ...           max_val: 20.0;
        ...           }
        ...         }
        ...       }
        ...       rnn1 {
        ...         rnn_type: LSTM;
        ...         hidden_size: 1152;
        ...         num_layers: 2;
        ...         bias: true;
        ...         bidirectional: false;
        ...         forget_gate_bias: {value: 1.0}
        ...             }
        ...
        ...       fc2 {
        ...       num_hidden_layers: 1;
        ...       hidden_size: 1152;
        ...       dropout: {value: 0.25}
        ...       activation {
        ...         hardtanh {
        ...           min_val: 0.0;
        ...           max_val: 20.0;
        ...           }
        ...         }
        ...       }
        ...     }
        ...     transducer_predict_net {
        ...       pred_nn {
        ...         rnn {
        ...           rnn_type: LSTM;
        ...           hidden_size: 256;
        ...           num_layers: 2;
        ...           bias: true;
        ...           bidirectional: false;
        ...           forget_gate_bias: {value: 1.0}
        ...         }
        ...       }
        ...     }
        ...     transducer_joint_net {
        ...       fc {
        ...         num_hidden_layers: 1;
        ...         hidden_size: 512;
        ...         dropout: {value: 0.25}
        ...         activation {
        ...           hardtanh {
        ...             min_val: 0.0;
        ...             max_val: 20.0;
        ...           }
        ...         }
        ...       }
        ...     }
        ... '''
        >>> cfg = text_format.Merge(cfg_txt, transducer_pb2.Transducer())
        >>> build(cfg, input_features=80, input_channels=5, vocab_size=28)
        Transducer(
          (encode): RNNTEncoder(
            (fc1): FullyConnected(
              (fully_connected): Sequential(
                (0): Linear(in_features=400, out_features=1152, bias=True)
                (1): Hardtanh(min_val=0.0, max_val=20.0)
                (2): Dropout(p=0.25, inplace=False)
                (3): Linear(in_features=1152, out_features=1152, bias=True)
              )
            )
            (rnn1): RNN(
              (rnn): LSTM(1152, 1152, num_layers=2)
            )
            (fc2): FullyConnected(
              (fully_connected): Sequential(
                (0): Linear(in_features=1152, out_features=1152, bias=True)
                (1): Hardtanh(min_val=0.0, max_val=20.0)
                (2): Dropout(p=0.25, inplace=False)
                (3): Linear(in_features=1152, out_features=512, bias=True)
              )
            )
          )
          (predict_net): RNNTPredictNet(
            (embedding): Embedding(28, 256)
            (pred_nn): RNN(
              (rnn): LSTM(256, 256, num_layers=2, batch_first=True)
            )
          )
          (joint_net): RNNTJointNet(
            (fc): FullyConnected(
              (fully_connected): Sequential(
                (0): Linear(in_features=768, out_features=512, bias=True)
                (1): Hardtanh(min_val=0.0, max_val=20.0)
                (2): Dropout(p=0.25, inplace=False)
                (3): Linear(in_features=512, out_features=29, bias=True)
              )
            )
          )
        )

    """
    # encoder output size is only required for build_transducer_enc_cfg if fc2
    # layer is present
    # (else it is necessarily specified by the encoder's rnn hidden size).
    # If fc2 layer *is* present, output size will be equal to joint
    # network hidden size:
    out_enc_size = None
    if (
        transducer_cfg.transducer_encoder.HasField("fc2")
        and transducer_cfg.transducer_joint_net.fc.hidden_size > 0
    ):
        out_enc_size = transducer_cfg.transducer_joint_net.fc.hidden_size
    encoder, encoder_out = build_transducer_enc_cfg(
        transducer_cfg.transducer_encoder,
        input_features * input_channels,
        output_features=out_enc_size,
    )
    if out_enc_size is not None:
        assert encoder_out == out_enc_size

    predict_net, predict_net_out = build_transducer_predict_net(
        transducer_cfg.transducer_predict_net, vocab_size
    )

    joint_in_dim = encoder_out + predict_net_out  # features are concatenated

    joint_net = build_joint_net(
        transducer_cfg.transducer_joint_net,
        input_features=joint_in_dim,
        output_features=vocab_size + 1,
    )

    return Transducer(
        encoder=encoder, predict_net=predict_net, joint_net=joint_net
    )


def build_transducer_enc_cfg(
    transducer_enc_cfg: transducer_encoder_pb2.TransducerEncoder,
    input_features: int,
    output_features: Optional[int] = None,
) -> Tuple[torch.nn.Module, int]:
    """Returns a transducer encoder based on the config.

    Args:
        transducer_enc_cfg: An ``TransducerEncoder`` protobuf object containing
            the config for the desired :py:class:`torch.nn.Module`.

        input_features: The number of features for the input.

        output_features: The number of output features of the encoder if
            ``transducer_enc_cfg.HasField('fc2')``. Otherwise, must have
            ``output_features=None`` since the output size will be equal to
            the hidden size of ``rnn1``.

    Returns:
        A Tuple where the first element is a :py:class:`torch.nn.Module`
        based on the config and the second element is the encoder output
        feature size. See :py:class:`Transducer` docstring for description of
        the encoder API.

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
        ...             transducer_encoder_pb2.TransducerEncoder()
        ... )
        >>> encoder, out = build_transducer_enc_cfg(cfg, input_features=400)
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
        >>> out
        576
    """
    if not transducer_enc_cfg.HasField("fc2"):
        assert output_features is None
    elif output_features is not None:
        assert (
            output_features > 0
        ), f"encoder output_features must be > 0 \
            but output_features={output_features}"

    # maybe add fc1:
    fc1: Optional[torch.nn.Module] = None

    if transducer_enc_cfg.HasField("fc1"):
        out_fc1 = transducer_enc_cfg.rnn1.hidden_size
        fc1 = build_fully_connected(
            transducer_enc_cfg.fc1,
            input_features=input_features,
            output_features=out_fc1,
        )
        input_features = out_fc1

    rnn1, rnn_out_features = build_rnn(transducer_enc_cfg.rnn1, input_features)
    # maybe add fc2:
    fc2: Optional[torch.nn.Module] = None
    if transducer_enc_cfg.HasField("fc2"):
        if output_features is None:
            # Halves feature size if possible:
            output_features = rnn_out_features // 2
            output_features = output_features if output_features > 0 else 1

        fc2 = build_fully_connected(
            transducer_enc_cfg.fc2,
            input_features=rnn_out_features,
            output_features=output_features,
        )
    else:
        output_features = rnn_out_features

    encoder = RNNTEncoder(rnn1=rnn1, fc1=fc1, fc2=fc2)

    return encoder, output_features


def build_transducer_predict_net(
    predict_net_cfg: transducer_predict_net_pb2.TransducerPredictNet,
    input_features: int,
) -> Tuple[torch.nn.Module, int]:
    """Returns a Transducer predict net based on the config.

    Currently only supports prediction network variant in which ``pred_nn`` is
    an RNN.

    Args:
        predict_net_cfg: a ``TransducerPredictNet`` protobuf object
            containing the config for the desired :py:class:`torch.nn.Module`.

        input_features: The input feature size.

    Returns:
        A Tuple where the first element is a :py:class:`torch.nn.Module`
        based on the config and the second element is the prediction output
        feature size. See :py:class:`Transducer` docstring for description of
        the prediction net API.
    """
    if not predict_net_cfg.pred_nn.HasField("rnn"):
        raise NotImplementedError(
            "Non rnn-based prediction network not supported."
        )
    # can get embedding dimension from the pred_nn config
    hidden_size = predict_net_cfg.pred_nn.rnn.hidden_size
    embedding = nn.Embedding(input_features, embedding_dim=hidden_size)

    # pred_nn is a batch_first=True rnn
    pred_nn, predict_net_out = build_rnn(
        predict_net_cfg.pred_nn.rnn, hidden_size, batch_first=True
    )
    # Set hidden_size attribute
    pred_nn.hidden_size = hidden_size
    predict_net = RNNTPredictNet(embedding=embedding, pred_nn=pred_nn)
    return predict_net, predict_net_out


def build_joint_net(
    transducer_joint_net_cfg: transducer_joint_net_pb2.TransducerJointNet,
    input_features: int,
    output_features: int,
) -> Tuple[torch.nn.Module, int]:
    """Returns a Transducer joint net based on the config.

    Currently only supports joint network variant with a single fc layer.

    Args:
        transducer_joint_net_cfg: a ``TransducerJointNet`` protobuf object
            containing the config for the desired :py:class:`torch.nn.Module`.

        input_features: The input feature size.

        output_features: The output feature size.

    Returns:
        A :py:class:`torch.nn.Module` based on the config. See
        :py:class:`Transducer` docstring for description of the joint net API.
    """
    fc = build_fully_connected(
        transducer_joint_net_cfg.fc,
        input_features=input_features,
        output_features=output_features,
    )
    return RNNTJointNet(fc=fc)
