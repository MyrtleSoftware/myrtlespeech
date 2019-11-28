from myrtlespeech.model.transducer import Transducer
from myrtlespeech.post_process.transducer_greedy_decoder import (
    TransducerGreedyDecoder,
)
from myrtlespeech.protos import transducer_greedy_decoder_pb2


def build(
    decoder_cfg: transducer_greedy_decoder_pb2.TransducerGreedyDecoder,
    model: Transducer,
) -> TransducerGreedyDecoder:
    r"""Returns a :py:class:`TransducerGreedyDecoder` based on the config.


    Args:
        decoder_cfg: A ``TransducerGreedyDecoder`` protobuf object containing
            the config for the desired :py:class:`TransducerGreedyDecoder`.

        model: A :py:class:`Transducer` model to use during decoding. See the
            :py:class:`Transducer` docstring for more information.

    Returns:
        A :py:class:`TransducerGreedyDecoder` based on the config.

    Raises:
        :py:class:`ValueError` if ``max_symbols_per_step`` < 1.
    """
    if (
        not decoder_cfg.max_symbols_per_step == 0
    ):  # 0 is default value (i.e. not set)

        max_symbols_per_step = decoder_cfg.max_symbols_per_step
        if max_symbols_per_step < 1:
            raise ValueError(
                f"decoder_cfg.max_symbols_per_step = \
                {max_symbols_per_step} but this must be >= 1"
            )
    else:
        max_symbols_per_step = None

    return TransducerGreedyDecoder(
        blank_index=decoder_cfg.blank_index,
        model=model,
        max_symbols_per_step=max_symbols_per_step,
    )
