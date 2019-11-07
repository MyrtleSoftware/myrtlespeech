"""A minimal CLI for running a config.

Under heavy development! Lots of code here should be moved elsewhere and
tested.
"""
import argparse
import time
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


class ReportCTCDecoder(Callback):
    """TODO

    Args:
        ctc_decoder: decodes output to sequence of indices based on CTC

        alphabet: converts sequences of indices to sequences of symbols (strs)

        word_segmentor: groups sequences of symbols into sequences of words
    """

    def __init__(self, ctc_decoder, alphabet, word_segmentor):
        self.ctc_decoder = ctc_decoder
        self.alphabet = alphabet
        self.word_segmentor = word_segmentor

    def _reset(self, **kwargs):
        kwargs["reports"][self.ctc_decoder.__class__.__name__] = {
            "wer": -1.0,
            "transcripts": [],
        }
        self.distances = []
        self.lengths = []

    def on_train_begin(self, **kwargs):
        self._reset(**kwargs)

    def on_epoch_begin(self, **kwargs):
        self._reset(**kwargs)

    def _process(self, sentence: List[int]) -> List[str]:
        symbols = self.alphabet.get_symbols(sentence)
        return self.word_segmentor(symbols)

    def on_batch_end(self, **kwargs):
        if self.training:
            return
        transcripts = kwargs["reports"][self.ctc_decoder.__class__.__name__][
            "transcripts"
        ]

        targets = kwargs["last_target"][0]
        target_lens = kwargs["last_target"][1]

        acts = self.ctc_decoder(*kwargs["last_output"])
        for act, target, target_len in zip(acts, targets, target_lens):
            act = self._process(act)
            exp = self._process([int(e) for e in target[:target_len]])

            transcripts.append((act, exp))

            distance = levenshtein(act, exp)
            self.distances.append(distance)
            self.lengths.append(len(exp))

    def on_epoch_end(self, **kwargs):
        if self.training:
            return
        wer = float(sum(self.distances)) / sum(self.lengths) * 100
        kwargs["reports"][self.ctc_decoder.__class__.__name__]["wer"] = wer


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
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            self.writer.add_histogram(
                name.replace(".", "/") + "/grad",
                param.grad,
                global_step=kwargs["total_train_batches"],
            )

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
