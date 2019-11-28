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


class ReportDecoderBase(Callback):
    """Base class for reporting error rates (WERs and CERs).

    *Do not use this class directly.* When overriding the base class, you must
    define the following:
        `self.decoder_input_key` @property - this gives the kwargs key required
            to access the decoder input.

    Args:
        decoder: decodes output to sequence of indices.

        alphabet: converts sequences of indices to sequences of symbols (strs).

        word_segmentor: groups sequences of symbols into sequences of words.
            By default this splits words on the space symbol " ".

        eval_every: WER/CER is cacluated every `eval_every`th epoch. Default
            is 1.

        calc_quantities: Iterable of strings of error quantities to calculate.
            The strings can take the following values:

                'wer': Word-error rate is calculated.

                'cer': Character-error rate is calculated.

            `calc_quantities` defaults to ('cer', 'wer').
    """

    def __init__(
        self,
        decoder,
        alphabet,
        word_segmentor=WordSegmentor(" "),
        eval_every=1,
        calc_quantities=("cer", "wer"),
    ):
        self.decoder = decoder
        self.alphabet = alphabet
        self.word_segmentor = word_segmentor
        self.eval_every = eval_every

        for idx, error_rate in enumerate(calc_quantities):
            assert error_rate in [
                "cer",
                "wer",
            ], f"calc_quantities[{idx}]={error_rate} is not in ['cer', 'wer']."
        assert (
            list(set(calc_quantities)).sort() == list(calc_quantities).sort()
        ), f"Repeated error rate in {calc_quantities}"
        assert len(calc_quantities) > 0, "calc_quantities cannot be empty!."
        self.calc_quantities = list(calc_quantities)

    def _reset(self, **kwargs):
        decoder_report = {quant: -1 for quant in self.calc_quantities}
        decoder_report["transcripts"] = []
        kwargs["reports"][self.decoder.__class__.__name__] = decoder_report
        self.distances = {quant: [] for quant in self.calc_quantities}
        self.lengths = {quant: [] for quant in self.calc_quantities}

    def on_train_begin(self, **kwargs):
        self._reset(**kwargs)

    def on_epoch_begin(self, **kwargs):
        self._reset(**kwargs)

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
            act_chars = self.alphabet.get_symbols(act)
            exp_chars = self.alphabet.get_symbols(
                [int(e) for e in target[:target_len]]
            )
            act_words = self.word_segmentor(act_chars)
            exp_words = self.word_segmentor(exp_chars)
            transcripts.append((act_words, exp_words))

            for error_rate in self.calc_quantities:
                if error_rate == "wer":
                    distance = levenshtein(act_words, exp_words)
                    self.lengths[error_rate].append(len(exp_words))
                elif error_rate == "cer":
                    distance = levenshtein(act_chars, exp_chars)
                    self.lengths[error_rate].append(len(exp_chars))
                else:
                    raise ValueError("error_rate is not in ['cer', 'wer'].")

                self.distances[error_rate].append(distance)

    def on_epoch_end(self, **kwargs):
        if self.training:
            return
        for error_rate in self.calc_quantities:
            lengths = sum(self.lengths[error_rate])
            if lengths != 0:
                err = (
                    float(sum(self.distances[error_rate]))
                    / sum(self.lengths[error_rate])
                    * 100
                )
            else:
                warnings.warn(
                    "Total length of input sequences == 0. \
                    Cannot calculate WER/CER"
                )
                err = -1  # return infeasible value

            kwargs["reports"][self.decoder.__class__.__name__][
                error_rate
            ] = err


class ReportCTCDecoder(ReportDecoderBase):
    """CTC Decoder Callback

    Args:
        See :py:class`ReportDecoderBase`.
    """

    def __init__(
        self,
        ctc_decoder,
        alphabet,
        word_segmentor,
        eval_every=1,
        calc_quantities=("cer", "wer"),
    ):
        super().__init__(
            ctc_decoder, alphabet, word_segmentor, eval_every, calc_quantities
        )

    @property
    def decoder_input_key(self):
        return "last_output"


class ReportRNNTDecoder(ReportDecoderBase):
    """RNNT Decoder Callback.

    Args:
        skip_first_epoch: bool. Default = False. If True, the first eval epoch
            is skipped. This is useful as the decoding is *very* slow with an
            un-trained model (i.e. inference is considerably faster when the
            model is more confident in its predictions).

        See :py:class`ReportDecoderBase` for other args.
    """

    def __init__(
        self,
        rnnt_decoder,
        alphabet,
        word_segmentor=WordSegmentor(" "),
        eval_every=1,
        calc_quantities=("cer", "wer"),
        skip_first_epoch=False,
    ):
        super().__init__(
            rnnt_decoder, alphabet, word_segmentor, eval_every, calc_quantities
        )
        self.skip_first_epoch = skip_first_epoch

    def on_batch_end(self, **kwargs):
        r"""Performs error-rate calculation."""
        if self.skip_first_epoch and kwargs["epoch"] == 0:
            return
        # The `RNNTTraining` callback sets kwargs["last_input"] = (x, y) as
        # ground truth values y are required for the Transducer forward pass
        # but the decoders *should not* have access to ground truth labels
        # and hence kwargs["last_input"] is updated here:
        x, y = kwargs["last_input"]
        kwargs["last_input"] = x
        return super().on_batch_end(**kwargs)

    @property
    def decoder_input_key(self):
        return "last_input"  # i.e. we can't use the rnnt output during
        # decoding as this was computed using ground-truth labels!


class TensorBoardLogger(ModelCallback):
    r"""Enables TensorBoard logging.

    .. note::
        If :py:class:`MixedPrecision` is used then this should appear earlier
        in the list of callbacks (i.e. have a lower index) as the
        :py:class:`MixedPrecision` callback shold rescale the losses *after*
        logging.

    Args:
        path: A path-like object that represents the tensorboard log_dir
            location.
        model: A :py:class:`torch.nn.Module`.
        histograms: If True, gradient and parameter histograms are saved.
            Defaults to False as this adds *substantial* overhead.
    """

    def __init__(
        self,
        path: Union[str, Path],
        model: torch.nn.Module,
        histograms: bool = False,
    ):
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
        # If MixedPrecision callback is used, this callback will thrown an
        # exception for the batches for which the loss is rescaled (as there
        # are no gradients). Hence use a try/except:
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
            pass

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
                    f"{stage}/{k}",
                    v,
                    global_step=kwargs["total_train_batches"],
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
                seq_to_seq.post_process,
                seq_to_seq.alphabet,
                WordSegmentor(" "),
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
