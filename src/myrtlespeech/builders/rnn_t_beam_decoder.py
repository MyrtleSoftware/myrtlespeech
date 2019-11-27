from myrtlespeech.model.transducer import Transducer
from myrtlespeech.post_process.rnn_t_beam_decoder import RNNTBeamDecoder
from myrtlespeech.protos import rnn_t_beam_decoder_pb2


def build(
    rnn_t_beam_decoder_cfg: rnn_t_beam_decoder_pb2.RNNTBeamDecoder,
    model: Transducer,
) -> RNNTBeamDecoder:
    """Returns a :py:class:`RNNTBeamDecoder` based on the config.

    Args:
        rnn_t_beam_decoder_cfg: A ``RNNTBeamDecoder`` protobuf object
            containing the config for the desired :py:class:`RNNTBeamDecoder`.

        model: A :py:class:`myrtlespeech.model.transducer.Transducer` model
            to use during decoding. See the
            :py:class:`myrtlespeech.model.transducer.Transducer`
            docstring for more information.

    Returns:
        A :py:class:`RNNTBeamDecoder` based on the config.

    Raises:
        :py:class:`ValueError` if ``max_symbols_per_step`` < 1.
    """

    if (
        not rnn_t_beam_decoder_cfg.max_symbols_per_step == 0
    ):  # 0 is default value (i.e. not set)

        max_symbols_per_step = rnn_t_beam_decoder_cfg.max_symbols_per_step
        if max_symbols_per_step < 1:
            raise ValueError(
                f"rnn_t_beam_decoder_cfg.max_symbols_per_step \
                = {max_symbols_per_step} but this must be >= 1"
            )
    else:
        max_symbols_per_step = None

    kwargs = {
        "blank_index": rnn_t_beam_decoder_cfg.blank_index,
        "model": model,
        "max_symbols_per_step": max_symbols_per_step,
    }

    if not rnn_t_beam_decoder_cfg.beam_width == 0:  # default is 0
        kwargs["beam_width"] = rnn_t_beam_decoder_cfg.beam_width

    kwargs[
        "length_norm"
    ] = rnn_t_beam_decoder_cfg.length_norm  # default is False

    return RNNTBeamDecoder(**kwargs)
