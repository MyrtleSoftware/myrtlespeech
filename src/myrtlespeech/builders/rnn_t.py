from typing import Tuple

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
    rnn_t_cfg: rnn_t_pb2.RNNT, input_features: int, vocab_size: int
) -> RNNT:
    """
    TODO
    """
    encoder, encoder_out = build_rnnt_enc(
        rnn_t_cfg.rnn_t_encoder, input_features
    )

    ##decoder/prediction network
    # can get embedding dims from the rnnt
    embedding = nn.Embedding(vocab_size, rnn_t_cfg.dec_rnn.hidden_size)
    dec_rnn, prediction_out = build_rnn(rnn_t_cfg.dec_rnn, vocab_size)

    ##joint
    fc_in_dim = encoder_out + prediction_out  # features are concatenated

    fully_connected = build_fully_connected(
        rnn_t_cfg.fully_connected,
        input_features=fc_in_dim,
        output_features=vocab_size + 1,
    )

    return RNNT(encoder, embedding, dec_rnn, fully_connected)


def build_rnnt_enc(
    rnn_t_enc: rnn_t_encoder_pb2.RNNTEncoder, input_features: int
) -> Tuple[RNNTEncoder, int]:
    """
    TODO

    """
    rnn1, rnn1_out_features = build_rnn(rnn_t_enc.rnn1, input_features)

    if rnn_t_enc.time_reduction_factor == 0:  # default value (i.e. not set)
        assert rnn_t_enc.HasField("rnn2") is False
        encoder = RNNTEncoder(rnn1)

        encoder_out_features = rnn1_out_features
    else:
        time_reduction_factor = rnn_t_enc.time_reduction_factor

        assert (
            time_reduction_factor > 1
        ), "time_reduction_factor must be an integer > 1 but is = {time_reduction_factor}"

        reduction = rnn_t_enc.time_reduction_factor

        time_reducer = Lambda(StackTime(reduction))

        rnnt_input_features = rnn1_out_features * reduction

        rnn2, encoder_out_features = build_rnn(
            rnn_t_enc.rnn2, rnnt_input_features
        )
        encoder = RNNTEncoder(rnn1, time_reducer, reduction, rnn2)

    return encoder, encoder_out_features
