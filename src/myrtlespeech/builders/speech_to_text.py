"""Builds an :py:class:`.SpeechToText` model from a configuration."""
from typing import Callable, List, Tuple

from myrtlespeech.builders.ctc_beam_decoder import (
    build as build_ctc_beam_decoder,
)
from myrtlespeech.builders.ctc_loss import build as build_ctc_loss
from myrtlespeech.builders.encoder_decoder import build as build_encoder_decoder
from myrtlespeech.builders.pre_process_step import (
    build as build_pre_process_step,
)
from myrtlespeech.data.alphabet import Alphabet
from myrtlespeech.data.preprocess import MFCC
from myrtlespeech.model.speech_to_text import SpeechToText
from myrtlespeech.post_process.ctc_greedy_decoder import CTCGreedyDecoder
from myrtlespeech.protos import speech_to_text_pb2


def build(stt_cfg: speech_to_text_pb2.SpeechToText,) -> SpeechToText:
    """Returns a :py:class:`.SpeechToText` model based on the config.

    .. note::

        Does not verify that the configured `.SpeechToText` model is valid
        (i.e. whether the sequence of ``pre_processing_step``s is valid etc).

    Args:
        stt_cfg: A ``SpeechToText`` protobuf object containing the config for
            the desired :py:class:`.SpeechToText`.

    Returns:
        An :py:class:`.SpeechToText` based on the config.

    Raises:
        :py:class:`ValueError`: On invalid configuration.
    """
    if not isinstance(stt_cfg, speech_to_text_pb2.SpeechToText):
        raise ValueError(
            f"type(stt_cfg)={type(stt_cfg)} and not SpeechToText protobuf"
        )

    alphabet = Alphabet(stt_cfg.alphabet)

    # preprocessing
    input_channels = None
    input_features = 1
    pre_process_steps: List[Tuple[Callable, bool]] = []
    for step_cfg in stt_cfg.pre_process_step:
        step = build_pre_process_step(step_cfg)
        if isinstance(step[0], MFCC):
            input_features = step[0].numcep
        else:
            raise ValueError(f"unknown step={type(step)}")
        pre_process_steps.append(step)

    # model
    model_type = stt_cfg.WhichOneof("model")
    if model_type == "encoder_decoder":
        # if CNN present and pre-processing not created channel dim set it to 1
        cnn = stt_cfg.encoder_decoder.encoder.WhichOneof("supported_cnns")
        if cnn != "no_cnn" and input_channels is None:
            input_channels = 1

        model = build_encoder_decoder(
            encoder_decoder_cfg=stt_cfg.encoder_decoder,
            input_features=input_features,
            output_features=len(alphabet),
            input_channels=input_channels,
        )
    else:
        raise ValueError(f"model={model_type} not supported")

    # capture "blank_index"s in all CTC-based components and check all match
    ctc_blank_indices: List[int] = []

    # loss
    loss_type = stt_cfg.WhichOneof("loss")
    if loss_type == "ctc_loss":
        blank_index = stt_cfg.ctc_loss.blank_index
        ctc_blank_indices.append(blank_index)
        if not (0 <= blank_index <= max(0, len(alphabet) - 1)):
            raise ValueError(
                f"ctc_loss.blank_index={blank_index} must be in "
                f"[0, {max(0, len(alphabet) - 1)}]"
            )
        loss = build_ctc_loss(stt_cfg.ctc_loss)
    else:
        raise ValueError(f"loss={loss_type} not supported")

    # post processing
    post_process_type = stt_cfg.WhichOneof("post_process")
    if post_process_type == "ctc_greedy_decoder":
        blank_index = stt_cfg.ctc_greedy_decoder.blank_index
        ctc_blank_indices.append(blank_index)
        if not (0 <= blank_index <= max(0, len(alphabet) - 1)):
            raise ValueError(
                f"ctc_greedy_decoder.blank_index={blank_index} must be in "
                f"[0, {max(0, len(alphabet) - 1)}]"
            )
        post_process = CTCGreedyDecoder(blank_index=blank_index)
    elif post_process_type == "ctc_beam_decoder":
        blank_index = stt_cfg.ctc_beam_decoder.blank_index
        ctc_blank_indices.append(blank_index)
        if not (0 <= blank_index <= max(0, len(alphabet) - 1)):
            raise ValueError(
                f"ctc_beam_decoder.blank_index={blank_index} must be in "
                f"[0, {max(0, len(alphabet) - 1)}]"
            )
        if stt_cfg.ctc_beam_decoder.HasField("separator_index"):
            separator_index = stt_cfg.ctc_beam_decoder.separator_index.value
            if not (0 <= separator_index <= max(0, len(alphabet) - 1)):
                raise ValueError(
                    f"ctc_beam_decoder.separator_index.value={separator_index} "
                    f"[0, {max(0, len(alphabet) - 1)}]"
                )
        post_process = build_ctc_beam_decoder(stt_cfg.ctc_beam_decoder)
    else:
        raise ValueError(f"post_process={post_process_type} not supported")

    # check all "blank_index"s are equal
    if ctc_blank_indices and not len(set(ctc_blank_indices)) == 1:
        raise ValueError("all blank_index values of CTC components must match")

    stt = SpeechToText(
        alphabet=alphabet,
        model=model,
        loss=loss,
        pre_process_steps=pre_process_steps,
        post_process=post_process,
    )
    return stt
