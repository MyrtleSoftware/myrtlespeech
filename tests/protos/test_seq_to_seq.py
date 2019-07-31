from typing import Dict, Optional, Tuple, Union

import hypothesis.strategies as st

from myrtlespeech.protos import seq_to_seq_pb2
from myrtlespeech.protos import ctc_greedy_decoder_pb2
from tests.data.test_alphabet import random_alphabet
from tests.protos.test_ctc_loss import ctc_losses
from tests.protos.test_pre_process_step import pre_process_steps
from tests.protos.test_ctc_beam_decoder import ctc_beam_decoders
from tests.protos.test_encoder_decoder import encoder_decoders
from tests.protos.utils import all_fields_set


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def seq_to_seqs(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[seq_to_seq_pb2.SeqToSeq],
    st.SearchStrategy[Tuple[seq_to_seq_pb2.SeqToSeq, Dict]],
]:
    """Returns a SearchStrategy for SeqToSeq plus maybe the kwargs."""
    kwargs: Dict = {}
    kwargs["alphabet"] = "".join(draw(random_alphabet(min_size=2)).symbols)

    descript = seq_to_seq_pb2.SeqToSeq.DESCRIPTOR

    # model
    model_str = draw(
        st.sampled_from(
            [f.name for f in descript.oneofs_by_name["model"].fields]
        )
    )
    if model_str == "encoder_decoder":
        kwargs["encoder_decoder"] = draw(encoder_decoders(valid_only=True))
    else:
        raise ValueError(f"unknown model type {model_str}")

    # record CTC blank index to share between CTC components
    ctc_blank_index: Optional[int] = None

    # loss
    loss_str = draw(
        st.sampled_from(
            [f.name for f in descript.oneofs_by_name["loss"].fields]
        )
    )
    if loss_str == "ctc_loss":
        kwargs["ctc_loss"] = draw(
            ctc_losses(alphabet_len=len(kwargs["alphabet"]))
        )
        ctc_blank_index = kwargs["ctc_loss"].blank_index
    else:
        raise ValueError(f"unknown loss type {loss_str}")

    # preprocess step
    kwargs["pre_process_step"] = []
    if draw(st.booleans()):
        kwargs["pre_process_step"].append(draw(pre_process_steps()))

    # post process
    post_str = draw(
        st.sampled_from(
            [f.name for f in descript.oneofs_by_name["post_process"].fields]
        )
    )
    if post_str == "ctc_greedy_decoder":
        if ctc_blank_index is None:
            ctc_blank_index = draw(
                st.integers(0, max(0, len(kwargs["alphabet"]) - 1))
            )
        kwargs["ctc_greedy_decoder"] = ctc_greedy_decoder_pb2.CTCGreedyDecoder(
            blank_index=ctc_blank_index
        )
    elif post_str == "ctc_beam_decoder":
        beam_kwargs = {"alphabet_len": len(kwargs["alphabet"])}
        if ctc_blank_index is not None:
            beam_kwargs["blank_index"] = ctc_blank_index
        kwargs["ctc_beam_decoder"] = draw(ctc_beam_decoders(**beam_kwargs))
    else:
        raise ValueError(f"unknown post_process type {post_str}")

    # initialise and return
    all_fields_set(seq_to_seq_pb2.SeqToSeq, kwargs)
    speech_to_text = seq_to_seq_pb2.SeqToSeq(  # type: ignore
        **kwargs
    )
    if not return_kwargs:
        return speech_to_text
    return speech_to_text, kwargs
