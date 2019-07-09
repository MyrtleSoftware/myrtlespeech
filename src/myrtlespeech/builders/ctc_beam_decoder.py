from myrtlespeech.builders.language_model import build as build_lm
from myrtlespeech.post_process.ctc_beam_decoder import CTCBeamDecoder
from myrtlespeech.protos import ctc_beam_decoder_pb2


def build(
    ctc_beam_decoder_cfg: ctc_beam_decoder_pb2.CTCBeamDecoder
) -> CTCBeamDecoder:
    """Returns a :py:class:`CTCBeamDecoder` based on the config.

    Args:
        ctc_beam_decoder_cfg: A ``CTCBeamDecoder`` protobuf object containing
            the config for the desired :py:class:`CTCBeamDecoder`.

    Returns:
        A :py:class:`CTCBeamDecoder` based on the config.

    Example:

        >>> from google.protobuf import text_format
        >>> ctc_beam_decoder_cfg_text = '''
        ... blank_index: 0;
        ... beam_width: 20;
        ... prune_threshold: 0.01;
        ... language_model {
        ...   no_lm { }
        ... }
        ... separator_index {
        ...   value: 1;
        ... }
        ... word_weight: 1.0;
        ... '''
        >>> ctc_beam_decoder_cfg = text_format.Merge(
        ...     ctc_beam_decoder_cfg_text,
        ...     ctc_beam_decoder_pb2.CTCBeamDecoder()
        ... )
        >>> build(ctc_beam_decoder_cfg)
        CTCBeamDecoder(blank_index=0, beam_width=20, prune_threshold=0.009999999776482582, language_model=None, lm_weight=None, separator_index=1, word_weight=1.0)
    """
    lm = build_lm(ctc_beam_decoder_cfg.language_model)

    separator_index = None
    if ctc_beam_decoder_cfg.HasField("separator_index"):
        separator_index = ctc_beam_decoder_cfg.separator_index.value

    lm_weight = None
    if ctc_beam_decoder_cfg.HasField("lm_weight"):
        lm_weight = ctc_beam_decoder_cfg.lm_weight.value

    ctc_beam_decoder = CTCBeamDecoder(
        blank_index=ctc_beam_decoder_cfg.blank_index,
        beam_width=ctc_beam_decoder_cfg.beam_width,
        prune_threshold=ctc_beam_decoder_cfg.prune_threshold,
        language_model=lm,
        lm_weight=lm_weight,
        separator_index=separator_index,
        word_weight=ctc_beam_decoder_cfg.word_weight,
    )

    return ctc_beam_decoder
