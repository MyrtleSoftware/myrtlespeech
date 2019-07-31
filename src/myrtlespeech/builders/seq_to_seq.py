"""Builds an :py:class:`.SeqToSeq` model from a configuration."""
from typing import Callable
from typing import List
from typing import Tuple

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
from myrtlespeech.model.seq_to_seq import SeqToSeq
from myrtlespeech.post_process.ctc_greedy_decoder import CTCGreedyDecoder
from myrtlespeech.protos import seq_to_seq_pb2


def build(
    s2s_cfg: seq_to_seq_pb2.SeqToSeq, seq_len_support: bool = False
) -> SeqToSeq:
    """Returns a :py:class:`.SeqToSeq` model based on the config.

    .. note::

        Does not verify that the configured `.SeqToSeq` model is valid (i.e.
        whether the sequence of ``pre_processing_step``s is valid etc).

    Args:
        s2s_cfg: A :py:class:`seq_to_seq_pb2.SeqToSeq` protobuf object
            containing the config for the desired :py:class:`.SeqToSeq`.

        seq_len_support: If :py:data:`True`, the
            :py:meth:`torch.nn.Module.forward` method of the returned
            :py:class:`.SeqToSeq` model must optionally accept a
            ``seq_lens`` kwarg.

    Returns:
        An :py:class:`.SeqToSeq` based on the config.

    Raises:
        :py:class:`ValueError`: On invalid configuration.
    """
    if not isinstance(s2s_cfg, seq_to_seq_pb2.SeqToSeq):
        raise ValueError(
            f"type(s2s_cfg)={type(s2s_cfg)} and not SeqToSeq protobuf"
        )

    alphabet = Alphabet(s2s_cfg.alphabet)

    # preprocessing
    input_channels = None
    input_features = 1
    pre_process_steps: List[Tuple[Callable, bool]] = [
        # ensure raw audio signal is in floating-point format
        # why? e.g. MFCC computation truncates when in integer format
        (lambda x: x.float(), False)
    ]
    for step_cfg in s2s_cfg.pre_process_step:
        step = build_pre_process_step(step_cfg)
        if isinstance(step[0], MFCC):
            input_features = step[0].numcep
        else:
            raise ValueError(f"unknown step={step[0]}")
        pre_process_steps.append(step)

    if input_channels is None:
        # data after all other steps has size [features, seq_len], convert to
        # [channels (1), features, seq_len]
        pre_process_steps.append((lambda x: x.unsqueeze(0), False))
        input_channels = 1

    # model
    model_type = s2s_cfg.WhichOneof("model")
    if model_type == "encoder_decoder":
        model = build_encoder_decoder(
            encoder_decoder_cfg=s2s_cfg.encoder_decoder,
            input_features=input_features,
            output_features=len(alphabet),
            input_channels=input_channels,
            seq_len_support=seq_len_support,
        )
    else:
        raise ValueError(f"model={model_type} not supported")

    # capture "blank_index"s in all CTC-based components and check all match
    ctc_blank_indices: List[int] = []

    # loss
    loss_type = s2s_cfg.WhichOneof("loss")
    if loss_type == "ctc_loss":
        blank_index = s2s_cfg.ctc_loss.blank_index
        ctc_blank_indices.append(blank_index)
        if not (0 <= blank_index <= max(0, len(alphabet) - 1)):
            raise ValueError(
                f"ctc_loss.blank_index={blank_index} must be in "
                f"[0, {max(0, len(alphabet) - 1)}]"
            )
        loss = build_ctc_loss(s2s_cfg.ctc_loss)
    else:
        raise ValueError(f"loss={loss_type} not supported")

    # post processing
    post_process_type = s2s_cfg.WhichOneof("post_process")
    if post_process_type == "ctc_greedy_decoder":
        blank_index = s2s_cfg.ctc_greedy_decoder.blank_index
        ctc_blank_indices.append(blank_index)
        if not (0 <= blank_index <= max(0, len(alphabet) - 1)):
            raise ValueError(
                f"ctc_greedy_decoder.blank_index={blank_index} must be in "
                f"[0, {max(0, len(alphabet) - 1)}]"
            )
        post_process = CTCGreedyDecoder(blank_index=blank_index)
    elif post_process_type == "ctc_beam_decoder":
        blank_index = s2s_cfg.ctc_beam_decoder.blank_index
        ctc_blank_indices.append(blank_index)
        if not (0 <= blank_index <= max(0, len(alphabet) - 1)):
            raise ValueError(
                f"ctc_beam_decoder.blank_index={blank_index} must be in "
                f"[0, {max(0, len(alphabet) - 1)}]"
            )
        if s2s_cfg.ctc_beam_decoder.HasField("separator_index"):
            separator_index = s2s_cfg.ctc_beam_decoder.separator_index.value
            if not (0 <= separator_index <= max(0, len(alphabet) - 1)):
                raise ValueError(
                    f"ctc_beam_decoder.separator_index.value={separator_index} "
                    f"[0, {max(0, len(alphabet) - 1)}]"
                )
        post_process = build_ctc_beam_decoder(s2s_cfg.ctc_beam_decoder)
    else:
        raise ValueError(f"post_process={post_process_type} not supported")

    # check all "blank_index"s are equal
    if ctc_blank_indices and not len(set(ctc_blank_indices)) == 1:
        raise ValueError("all blank_index values of CTC components must match")

    s2s = SeqToSeq(
        alphabet=alphabet,
        model=model,
        loss=loss,
        pre_process_steps=pre_process_steps,
        post_process=post_process,
    )
    return s2s
