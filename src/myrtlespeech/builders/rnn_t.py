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
    """
    TODO
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
    """
    TODO

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
