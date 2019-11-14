"""A minimal CLI for running a config.

Under heavy development! Lots of code here should be moved elsewhere and
tested.
"""
import argparse
import time
import warnings
from pathlib import Path
from typing import List
from typing import Union

import torch
from google.protobuf import text_format  # type: ignore
from myrtlespeech.builders.task_config import build
from myrtlespeech.post_process.utils import levenshtein
from myrtlespeech.protos import task_config_pb2
from myrtlespeech.run.callbacks.callback import Callback
from myrtlespeech.run.callbacks.callback import ModelCallback
from myrtlespeech.run.callbacks.csv_logger import CSVLogger
from myrtlespeech.run.callbacks.mixed_precision import MixedPrecision
from myrtlespeech.run.callbacks.report_mean_batch_loss import (
    ReportMeanBatchLoss,
)
from myrtlespeech.run.callbacks.stop_epoch_after import StopEpochAfter
from myrtlespeech.run.train import fit
from torch.utils.tensorboard import SummaryWriter


class WordSegmentor:
    """TODO"""

    def __init__(self, separator: str):
        self.separator = separator

    def __call__(self, sentence: List[str]) -> List[str]:
        new_sentence = []
        word = []  # type: ignore
        for symb in sentence:
            if symb == self.separator:
                if word:
                    new_sentence.append("".join(word))
                    word = []
            else:
                word.append(symb)
        if word:
            new_sentence.append("".join(word))
        return new_sentence


class ReportDecoderWERBase(Callback):
    """Base class for reporting WERs.

    *Do not use this class directly.* When overriding the base class, you must
    define the following:
        `self._process_sentence()` method
        `self.decoder_input_key` @property - this gives the kwargs key required
            to access the decoder input.

    Args:
        decoder: decodes output to sequence of indices.

        alphabet: converts sequences of indices to sequences of symbols (strs).

        eval_every: WER is cacluated every `eval_every`th epoch. Default is 1.
    """

    def __init__(self, decoder, alphabet, eval_every=1):
        self.decoder = decoder
        self.alphabet = alphabet
        self.eval_every = eval_every

    def _reset(self, **kwargs):
        kwargs["reports"][self.decoder.__class__.__name__] = {
            "wer": -1.0,
            "transcripts": [],
        }
        self.distances = []
        self.lengths = []

    def on_train_begin(self, **kwargs):
        self._reset(**kwargs)

    def on_epoch_begin(self, **kwargs):
        self._reset(**kwargs)

    def _process_sentence(self, sentence: List[int]) -> List[str]:
        """Method to convert list of indexes to list of characters.
        This must be overidden by the inherited class"""

        raise NotImplementedError(
            "Must implement this method according to the given decoder type"
        )

    @property
    def decoder_input_key(self):
        raise NotImplementedError(
            "Must define `self.decoder_input_key` @property"
        )

    def on_batch_end(self, **kwargs):
        if self.training or kwargs["epoch"] % self.eval_every != 0:
            return
        transcripts = kwargs["reports"][self.decoder.__class__.__name__][
            "transcripts"
        ]

        targets = kwargs["last_target"][0]
        target_lens = kwargs["last_target"][1]
        acts = self.decoder(*kwargs[self.decoder_input_key])
        for act, target, target_len in zip(acts, targets, target_lens):
            act = self._process_sentence(act)
            exp = self._process_sentence([int(e) for e in target[:target_len]])

            transcripts.append((act, exp))

            distance = levenshtein(act, exp)
            self.distances.append(distance)
            self.lengths.append(len(exp))

    def on_epoch_end(self, **kwargs):
        if self.training:
            return
        lengths = sum(self.lengths)
        if lengths != 0:
            wer = float(sum(self.distances)) / sum(self.lengths) * 100
        else:
            warnings.warn(
                "Total length of input sequences == 0. Cannot calculate WER"
            )
            wer = -1  # return infeasible value
        kwargs["reports"][self.decoder.__class__.__name__]["wer"] = wer


class ReportCTCDecoder(ReportDecoderWERBase):
    """CTC Decoder Callback

    Args:
        ctc_decoder: decodes output to sequence of indices based on CTC

        alphabet: converts sequences of indices to sequences of symbols (strs)

        word_segmentor: groups sequences of symbols into sequences of words

        eval_every: WER is cacluated every `eval_every`th epoch. Default is 1.
    """

    def __init__(self, ctc_decoder, alphabet, word_segmentor, eval_every=1):
        super().__init__(ctc_decoder, alphabet, eval_every)
        self.word_segmentor = word_segmentor

    def _process_sentence(self, sentence: List[int]) -> List[str]:
        symbols = self.alphabet.get_symbols(sentence)
        return self.word_segmentor(symbols)

    @property
    def decoder_input_key(self):
        return "last_output"


class ReportRNNTDecoder(ReportDecoderWERBase):
    """RNNT Decoder Callback.

    Args:
        rnnt_decoder: decodes output to sequence of indices based on CTC

        alphabet: converts sequences of indices to sequences of symbols (strs)

        eval_every: WER is cacluated every `eval_every`th epoch. Default is 1.

        skip_first_epoch: bool. Default = False. If True, the first eval epoch
            is skipped. This is useful as the decoding is *very* slow with an
            un-trained model (i.e. inference is considerably faster when the
            model is more confident in its predictions)

    """

    def __init__(
        self, rnnt_decoder, alphabet, eval_every=1, skip_first_epoch=False
    ):
        super().__init__(rnnt_decoder, alphabet, eval_every)
        self.skip_first_epoch = skip_first_epoch

    def _process_sentence(self, sentence: List[int]) -> List[str]:
        return self.alphabet.get_symbols(sentence)

    def on_batch_end(self, **kwargs):
        if self.skip_first_epoch and kwargs["epoch"] == 0:
            return
        return super().on_batch_end(**kwargs)

    @property
    def decoder_input_key(self):
        return "last_input"  # i.e. we can't use the rnnt output during
        # decoding as this was computed using
        # ground-truth labels!


class TensorBoardLogger(ModelCallback):
    """TODO"""

    def __init__(self, path: Union[str, Path], model, histograms=False):
        super().__init__(model)
        self.writer = SummaryWriter(log_dir=str(path))
        self.histograms = histograms

    def on_backward_begin(self, **kwargs):
        stage = "train" if self.training else "eval"
        self.writer.add_scalar(
            f"{stage}/loss",
            kwargs["last_loss"].item(),
            global_step=kwargs["total_train_batches"],
        )

    def on_step_end(self, **kwargs):
        if not self.training or not self.histograms:
            return
        # when using MixedPrecision, if loss is rescaled, there will be no
        # gradients. Therefore put the histogram logging in a try/except:
        try:
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                self.writer.add_histogram(
                    name.replace(".", "/") + "/grad",
                    param.grad,
                    global_step=kwargs["total_train_batches"],
                )
        except ValueError:
            return

    def on_batch_end(self, **kwargs):
        if not self.training or not self.histograms:
            return
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(
                name.replace(".", "/"),
                param,
                global_step=kwargs["total_train_batches"],
            )

    def on_epoch_end(self, **kwargs):
        stage = "train" if self.training else "eval"
        for k, v in kwargs["reports"].items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(
                    f"{stage}/{k}", v, global_step=kwargs["total_train_batches"]
                )
            elif isinstance(v, dict):
                for k_p, v_p in v.items():
                    if isinstance(v_p, (int, float)):
                        self.writer.add_scalar(
                            f"{stage}/{k}/{k_p}",
                            v_p,
                            global_step=kwargs["total_train_batches"],
                        )

    def on_train_end(self, **kwargs):
        self.writer.close()


class Saver(ModelCallback):
    """TODO"""

    def __init__(self, log_dir: Union[str, Path], *args, **kwargs):
        self.log_dir = Path(log_dir)
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, **kwargs):
        if not self.training:
            return
        torch.save(
            self.model.state_dict(),
            str(self.log_dir.joinpath(f"state_dict_{kwargs['epoch']}.pt")),
        )


def parse(config_path: str) -> task_config_pb2.TaskConfig:
    """TODO"""
    with open(config_path) as f:
        task_config = text_format.Merge(f.read(), task_config_pb2.TaskConfig())
    return task_config


def run() -> None:
    # parse args
    parser = argparse.ArgumentParser(description="Run myrtlespeech")
    parser.add_argument("config")
    parser.add_argument("--log_dir")
    parser.add_argument("--disable_mixed_precision", action="store_true")
    parser.add_argument("--stop_epoch_after", type=int)
    args = parser.parse_args()

    # setup log directory
    log_dir = Path(args.log_dir or ".").joinpath(str(time.time()))
    log_dir.mkdir(parents=True)

    # setup Python objects
    task_config = parse(args.config)
    seq_to_seq, epochs, train_loader, eval_loader = build(task_config)

    # initialize callbacks
    callbacks = []
    callbacks.extend(
        [
            ReportMeanBatchLoss(),
            ReportCTCDecoder(
                seq_to_seq.post_process, seq_to_seq.alphabet, WordSegmentor(" ")
            ),
            TensorBoardLogger(log_dir, seq_to_seq.model, histograms=False),
        ]
    )

    if not args.disable_mixed_precision:
        callbacks.append(MixedPrecision(seq_to_seq, opt_level="O1"))

    if args.stop_epoch_after is not None:
        callbacks.append(StopEpochAfter(epoch_batches=args.stop_epoch_after))

    callbacks.extend(
        [CSVLogger(log_dir.joinpath("log.csv")), Saver(log_dir, seq_to_seq)]
    )

    # train and evaluate
    fit(
        seq_to_seq=seq_to_seq,
        epochs=epochs,
        train_loader=train_loader,
        eval_loader=eval_loader,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    run()
