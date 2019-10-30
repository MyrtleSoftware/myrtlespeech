from myrtlespeech.model.rnn_t import RNNT
from myrtlespeech.post_process.rnn_t_beam_decoder import RNNTBeamDecoder
from myrtlespeech.protos import rnn_t_beam_decoder_pb2


def build(
    rnn_t_beam_decoder_cfg: rnn_t_beam_decoder_pb2.RNNTBeamDecoder, model: RNNT
) -> RNNTBeamDecoder:
    """Returns a :py:class:`RNNTBeamDecoder` based on the config.

    Args:
        rnn_t_beam_decoder_cfg: A ``RNNTBeamDecoder`` protobuf object containing
            the config for the desired :py:class:`RNNTBeamDecoder`.
        model: A :py:class:`myrtlespeech.model.rnn_t.RNNT` model to use during decoding
            See the py:class:`myrtlespeech.model.rnn_t.RNNT` docstring for more information.


    Returns:
        A :py:class:`RNNTBeamDecoder` based on the config.

    """
    # uint32 beam_width = 2;
    #
    # bool length_norm = 3;
    #
    # uint32 max_symbols_per_step = 4; //if this is not provided there is no limit

    raise NotImplementedError("RNNTBeamDecoder builder is not implemented")
