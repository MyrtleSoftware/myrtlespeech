"""Builds an :py:class:`.SpeechToText` model from a protobuf configuration."""
from typing import Callable
from typing import List
from typing import Tuple

from myrtlespeech.builders.ctc_beam_decoder import (
    build as build_ctc_beam_decoder,
)
from myrtlespeech.builders.ctc_loss import build as build_ctc_loss
from myrtlespeech.builders.deep_speech_2 import build as build_deep_speech_2
from myrtlespeech.builders.pre_process_step import (
    build as build_pre_process_step,
)
from myrtlespeech.builders.rnn_t import build as build_rnn_t
from myrtlespeech.builders.rnn_t_loss import build as build_rnn_t_loss
from myrtlespeech.data.alphabet import Alphabet
from myrtlespeech.data.preprocess import AddContextFrames
from myrtlespeech.data.preprocess import Standardize
from myrtlespeech.model.cnn import Conv1dTo2d
from myrtlespeech.model.deep_speech_1 import DeepSpeech1
from myrtlespeech.model.speech_to_text import SpeechToText
from myrtlespeech.post_process.ctc_greedy_decoder import CTCGreedyDecoder
from myrtlespeech.protos import pre_process_step_pb2
from myrtlespeech.protos import speech_to_text_pb2
from myrtlespeech.run.stage import Stage
from torchaudio.transforms import MFCC

# from myrtlespeech.builders.rnn_t import build as build_rnn_t


def build(stt_cfg: speech_to_text_pb2.SpeechToText) -> SpeechToText:
    r"""Returns a :py:class:`.SpeechToText` model based on the ``stt_cfg``.

    .. note::

        Does not verify that the configured `.SpeechToText` model is valid
        (i.e.  whether the sequence of ``pre_processing_step``s is valid etc).

    Args:
        stt_cfg: A :py:class:`speech_to_text_pb2.SpeechToText` protobuf object
            containing the config for the desired :py:class:`.SpeechToText`.

    Returns:
        An :py:class:`.SpeechToText` based on the config.

    Raises:
        :py:class:`ValueError`: On invalid configuration.

    Example:
        >>> from google.protobuf import text_format
        >>> cfg_text = '''
        ... alphabet: "acgt_";
        ...
        ... pre_process_step {
        ...   stage: TRAIN_AND_EVAL;
        ...   mfcc {
        ...     n_mfcc: 80;
        ...     win_length: 400;
        ...     hop_length: 160;
        ...   }
        ... }
        ...
        ... pre_process_step {
        ...   stage: TRAIN_AND_EVAL;
        ...   standardize {
        ...   }
        ... }
        ...
        ... deep_speech_2 {
        ...   conv_block {
        ...     conv1d {
        ...       output_channels: 4;
        ...       kernel_time: 5;
        ...       stride_time: 2;
        ...       padding_mode: SAME;
        ...       bias: true;
        ...     }
        ...     activation {
        ...       hardtanh {
        ...         min_val: 0.0;
        ...         max_val: 20.0;
        ...       }
        ...     }
        ...   }
        ...
        ...   rnn {
        ...     rnn_type: LSTM;
        ...     hidden_size: 1024;
        ...     num_layers: 3;
        ...     bias: true;
        ...     bidirectional: true;
        ...     forget_gate_bias {
        ...       value: 1.0;
        ...     }
        ...   }
        ...
        ...   fully_connected {
        ...     num_hidden_layers: 1;
        ...     hidden_size: 1024;
        ...     activation {
        ...       hardtanh {
        ...         min_val: 0.0;
        ...         max_val: 20.0;
        ...       }
        ...     }
        ...   }
        ... }
        ...
        ... ctc_loss {
        ...   blank_index: 4;
        ...   reduction: SUM;
        ... }
        ...
        ... ctc_greedy_decoder {
        ...   blank_index: 4;
        ... }
        ... '''
        >>> cfg = text_format.Merge(
        ...     cfg_text,
        ...     speech_to_text_pb2.SpeechToText()
        ... )
        >>> build(cfg)
        SpeechToText(
          (alphabet): Alphabet(symbols=['a', 'c', 'g', 't', '_'])
          (model): DeepSpeech2(
            (cnn): Sequential(
              (0): Conv2dTo1d(seq_len_support=True)
              (1): MaskConv1d(80, 4, kernel_size=(5,), stride=(2,), padding_mode=PaddingMode.SAME)
              (2): SeqLenWrapper(
                (module): Hardtanh(min_val=0.0, max_val=20.0)
                (seq_lens_fn): Identity()
              )
              (3): Conv1dTo2d(seq_len_support=True)
            )
            (rnn): RNN(
              (rnn): LSTM(4, 1024, num_layers=3, bidirectional=True)
            )
            (fully_connected): FullyConnected(
              (fully_connected): Sequential(
                (0): Linear(in_features=2048, out_features=1024, bias=True)
                (1): Hardtanh(min_val=0.0, max_val=20.0)
                (2): Linear(in_features=1024, out_features=5, bias=True)
              )
            )
          )
          (loss): CTCLoss(
            (log_softmax): LogSoftmax()
            (ctc_loss): CTCLoss()
          )
          (post_process): CTCGreedyDecoder(blank_index=4)
        )
    """
    alphabet = Alphabet(list(stt_cfg.alphabet))

    # preprocessing
    pre_process_steps, input_features, input_channels = _build_pre_process_steps(
        stt_cfg.pre_process_step
    )

    # model
    model_type = stt_cfg.WhichOneof("supported_models")
    if model_type == "deep_speech_1":
        model = DeepSpeech1(
            in_features=input_channels * input_features,
            n_hidden=stt_cfg.deep_speech_1.n_hidden,
            out_features=len(alphabet),
            drop_prob=stt_cfg.deep_speech_1.drop_prob,
            relu_clip=stt_cfg.deep_speech_1.relu_clip,
            forget_gate_bias=stt_cfg.deep_speech_1.forget_gate_bias,
        )
    elif model_type == "deep_speech_2":
        model = build_deep_speech_2(
            deep_speech_2_cfg=stt_cfg.deep_speech_2,
            input_features=input_features,
            input_channels=input_channels,
            output_features=len(alphabet),
        )
    elif model_type == "rnn_t":
        model = build_rnn_t(
            rnn_t_cfg=stt_cfg.rnn_t,
            input_features=input_features,
            vocab_size=len(alphabet) - 1,  # i.e. excluding the blank symbol
        )
    else:
        raise ValueError(f"model={model_type} not supported")

    # capture "blank_index"s in all CTC/RNNT-based components and check all match
    blank_indices: List[int] = []

    # loss
    loss_type = stt_cfg.WhichOneof("supported_losses")
    if loss_type in ["ctc_loss", "rnn_t_loss"]:
        if loss_type == "ctc_loss":
            loss = build_ctc_loss(stt_cfg.ctc_loss)
            blank_index = stt_cfg.ctc_loss.blank_index
        elif loss_type == "rnn_t_loss":
            loss = build_rnn_t_loss(stt_cfg.rnn_t_loss)
            blank_index = stt_cfg.rnn_t_loss.blank_index
        else:
            raise ValueError(
                f"loss_type={loss_type} is not in ['ctc_loss', 'rnn_t_loss']. \
            Bug has been introduced: (this path should not execute)"
            )

        blank_indices.append(blank_index)
        if not (0 <= blank_index <= max(0, len(alphabet) - 1)):
            raise ValueError(
                f"{loss_type}.blank_index={blank_index} must be in "
                f"[0, {max(0, len(alphabet) - 1)}]"
            )
    else:
        raise ValueError(f"loss={loss_type} not supported")

    # post processing
    post_process_type = stt_cfg.WhichOneof("supported_post_processes")
    if post_process_type == "ctc_greedy_decoder":
        blank_index = stt_cfg.ctc_greedy_decoder.blank_index
        blank_indices.append(blank_index)
        if not (0 <= blank_index <= max(0, len(alphabet) - 1)):
            raise ValueError(
                f"ctc_greedy_decoder.blank_index={blank_index} must be in "
                f"[0, {max(0, len(alphabet) - 1)}]"
            )
        post_process = CTCGreedyDecoder(blank_index=blank_index)
    elif post_process_type == "ctc_beam_decoder":
        blank_index = stt_cfg.ctc_beam_decoder.blank_index
        blank_indices.append(blank_index)
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
    if blank_indices and not len(set(blank_indices)) == 1:
        raise ValueError(
            "all blank_index values of CTC/RNNT components must match"
        )

    stt = SpeechToText(
        alphabet=alphabet,
        model=model,
        loss=loss,
        pre_process_steps=pre_process_steps,
        post_process=post_process,
    )
    return stt


def _build_pre_process_steps(
    pre_process_step_cfg: List[pre_process_step_pb2.PreProcessStep]
) -> Tuple[List[Tuple[Callable, Stage]], int, int]:
    """Returns the preprocessing steps, features, and channels."""
    input_features = None
    input_channels = 1
    pre_process_steps: List[Tuple[Callable, Stage]] = []
    for step_cfg in pre_process_step_cfg:
        step = build_pre_process_step(step_cfg)
        if isinstance(step[0], MFCC):
            input_features = step[0].n_mfcc
        elif isinstance(step[0], Standardize):
            pass
        elif isinstance(step[0], AddContextFrames):
            input_channels = 2 * step[0].n_context + 1
        else:
            raise ValueError(f"unknown step={step[0]}")
        pre_process_steps.append(step)

    if input_features is None:
        pre_process_steps.append(
            (Conv1dTo2d(seq_len_support=False), Stage.TRAIN_AND_EVAL)
        )
        input_features = 1

    return pre_process_steps, input_features, input_channels
