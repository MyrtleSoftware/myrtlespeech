from myrtlespeech.model.rnn_t import RNNT
from myrtlespeech.protos import rnn_t_pb2


def build(
    rnn_t_cfg: rnn_t_pb2.RNNT,
    input_features: int,
    input_channels: int,
    vocab_size: int,
) -> RNNT:
    """TODO"""
    rnn_t = RNNT(
        in_features=input_channels * input_features,
        vocab_size=vocab_size,
        encoder_n_hidden=rnn_t_cfg.transcription.n_hidden,
        encoder_rnn_layers=rnn_t_cfg.transcription.rnn_layers,
        pred_n_hidden=rnn_t_cfg.prediction.n_hidden,
        pred_rnn_layers=rnn_t_cfg.prediction.rnn_layers,
        joint_n_hidden=rnn_t_cfg.joint.n_hidden
    )
    return rnn_t
