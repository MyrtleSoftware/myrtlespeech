from myrtlespeech.model.transducer import Transducer
from myrtlespeech.post_process.transducer_beam_decoder import (
    TransducerBeamDecoder,
)
from myrtlespeech.protos import transducer_beam_decoder_pb2


def build(
    decoder_cfg: transducer_beam_decoder_pb2.TransducerBeamDecoder,
    model: Transducer,
) -> TransducerBeamDecoder:
    """Returns a :py:class:`TransducerBeamDecoder` based on the config.

    Args:
        decoder_cfg: A ``TransducerBeamDecoder`` protobuf object
            containing the config for the desired
            :py:class:`TransducerBeamDecoder`.

        model: A :py:class:`myrtlespeech.model.transducer.Transducer` model
            to use during decoding. See the
            :py:class:`myrtlespeech.model.transducer.Transducer`
            docstring for more information.

    Returns:
        A :py:class:`TransducerBeamDecoder` based on the config.

    Raises:
        :py:class:`ValueError`: if ``max_symbols_per_step`` < 1.
    """

    if (
        not decoder_cfg.max_symbols_per_step == 0
    ):  # 0 is default value (i.e. not set)

        max_symbols_per_step = decoder_cfg.max_symbols_per_step
        if max_symbols_per_step < 1:
            raise ValueError(
                f"decoder_cfg.max_symbols_per_step \
                = {max_symbols_per_step} but this must be >= 1"
            )
    else:
        max_symbols_per_step = 100

    kwargs = {
        "blank_index": decoder_cfg.blank_index,
        "model": model,
        "max_symbols_per_step": max_symbols_per_step,
    }

    if not decoder_cfg.beam_width == 0:  # default is 0
        kwargs["beam_width"] = decoder_cfg.beam_width

    kwargs["length_norm"] = decoder_cfg.length_norm  # default is False

    return TransducerBeamDecoder(**kwargs)
