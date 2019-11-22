from myrtlespeech.model.rnn_t import RNNT
from myrtlespeech.post_process.rnn_t_greedy_decoder import RNNTGreedyDecoder
from myrtlespeech.protos import rnn_t_greedy_decoder_pb2


def build(
    rnn_t_greedy_decoder_cfg: rnn_t_greedy_decoder_pb2.RNNTGreedyDecoder,
    model: RNNT,
) -> RNNTGreedyDecoder:
    r"""Returns a :py:class:`RNNTGreedyDecoder` based on the config.

    Args:
        rnn_t_greedy_decoder_cfg: A ``RNNTGreedyDecoder`` protobuf object
            containing the config for the desired
            :py:class:`RNNTGreedyDecoder`.


        model: A :py:class:`myrtlespeech.model.rnn_t.RNNT` model to use during
            decoding. See the py:class:`myrtlespeech.model.rnn_t.RNNT`
            docstring for more information.

    Returns:
        A :py:class:`RNNTGreedyDecoder` based on the config.

    Raises:
        :py:class:`ValueError` if ``max_symbols_per_step`` < 1.
    """
    if (
        not rnn_t_greedy_decoder_cfg.max_symbols_per_step == 0
    ):  # 0 is default value (i.e. not set)

        max_symbols_per_step = rnn_t_greedy_decoder_cfg.max_symbols_per_step
        if max_symbols_per_step < 1:
            raise ValueError(
                f"rnn_t_greedy_decoder_cfg.max_symbols_per_step = \
                {max_symbols_per_step} but this must be >= 1"
            )
    else:
        max_symbols_per_step = None

    return RNNTGreedyDecoder(
        blank_index=rnn_t_greedy_decoder_cfg.blank_index,
        model=model,
        max_symbols_per_step=max_symbols_per_step,
    )
