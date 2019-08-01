"""Builds an :py:class:`.SpeechToText` model from a protobuf configuration."""
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
from myrtlespeech.model.speech_to_text import SpeechToText
from myrtlespeech.post_process.ctc_greedy_decoder import CTCGreedyDecoder
from myrtlespeech.protos import speech_to_text_pb2
from myrtlespeech.run.stage import Stage


def build(
    stt_cfg: speech_to_text_pb2.SpeechToText, seq_len_support: bool = False
) -> SpeechToText:
    """Returns a :py:class:`.SpeechToText` model based on the ``stt_cfg``.

    .. note::

        Does not verify that the configured `.SpeechToText` model is valid
        (i.e.  whether the sequence of ``pre_processing_step``s is valid etc).

    Args:
        stt_cfg: A :py:class:`speech_to_text_pb2.SpeechToText` protobuf object
            containing the config for the desired :py:class:`.SpeechToText`.

        seq_len_support: If :py:data:`True`, the
            :py:meth:`torch.nn.Module.forward` method of the returned
            :py:meth:`.SpeechToText.model` must optionally accept a
            ``seq_lens`` kwarg.

    Returns:
        An :py:class:`.SpeechToText` based on the config.

    Raises:
        :py:class:`ValueError`: On invalid configuration.

    Example:
        >>> from google.protobuf import text_format
        >>> cfg_text = '''
        ... alphabet: "_acgt";
        ... encoder_decoder {
        ...   encoder {
        ...     cnn_rnn_encoder {
        ...       no_cnn {
        ...       }
        ...       rnn {
        ...         rnn_type: LSTM;
        ...         hidden_size: 128;
        ...         num_layers: 2;
        ...         bias: true;
        ...         bidirectional: false;
        ...       }
        ...     }
        ...   }
        ...   decoder {
        ...     fully_connected {
        ...       num_hidden_layers: 0;
        ...     }
        ...   }
        ... }
        ... ctc_loss {
        ...   blank_index: 0;
        ...   reduction: MEAN;
        ... }
        ... ctc_greedy_decoder {
        ...   blank_index: 0;
        ... }
        ... '''
        >>> cfg = text_format.Merge(
        ...     cfg_text,
        ...     speech_to_text_pb2.SpeechToText()
        ... )
        >>> build(cfg)
        SpeechToText(
          (alphabet): Alphabet(symbols=['_', 'a', 'c', 'g', 't'])
          (model): EncoderDecoder(
            (encoder): CNNRNNEncoder(
              (cnn_to_rnn): Lambda()
              (rnn): RNN(
                (rnn): LSTM(1, 128, num_layers=2)
              )
            )
            (decoder): FullyConnected(
              (fully_connected): Linear(in_features=128, out_features=5, bias=True)
            )
          )
          (loss): CTCLoss(
            (log_softmax): LogSoftmax()
            (ctc_loss): CTCLoss()
          )
          (post_process): CTCGreedyDecoder(blank_index=0)
        )
    """
    alphabet = Alphabet(list(stt_cfg.alphabet))

    # preprocessing
    input_channels = None
    input_features = 1
    pre_process_steps: List[Tuple[Callable, Stage]] = [
        # ensure raw audio signal is in floating-point format
        # why? e.g. MFCC computation truncates when in integer format
        (lambda x: x.float(), Stage.TRAIN_AND_EVAL)
    ]
    for step_cfg in stt_cfg.pre_process_step:
        step = build_pre_process_step(step_cfg)
        if isinstance(step[0], MFCC):
            input_features = step[0].numcep
        else:
            raise ValueError(f"unknown step={step[0]}")
        pre_process_steps.append(step)

    if input_channels is None:
        # data after all other steps has size [features, seq_len], convert to
        # [channels (1), features, seq_len]
        pre_process_steps.append(
            (lambda x: x.unsqueeze(0), Stage.TRAIN_AND_EVAL)
        )
        input_channels = 1

    # model
    model_type = stt_cfg.WhichOneof("supported_models")
    if model_type == "encoder_decoder":
        model = build_encoder_decoder(
            encoder_decoder_cfg=stt_cfg.encoder_decoder,
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
    loss_type = stt_cfg.WhichOneof("supported_losses")
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
    post_process_type = stt_cfg.WhichOneof("supported_post_processes")
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
