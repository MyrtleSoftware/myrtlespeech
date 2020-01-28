"""A minimal CLI for running a config.

Under heavy development! Lots of code here should be moved elsewhere and
tested.
"""
import argparse
import os
import re
import time
import warnings
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
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
    """Saves ``model``'s ``state_dict`` each epoch.

    This callback will also attempt to load the ``state_dict``: from
    ``load_fp`` if present and otherwise will load the most recent
    ``state_dict`` in ``log_dir`` (if present) **according to filenames of
    the form ``state_dict_<EPOCH>.pt``**.

    Args:
        log_dir: A pathlike object giving the logging directory.

        load_fp: An Optional pathlike object specifying the filepath of the
            state_dict.

        model: A :py:class:`torch.nn.Module` to be saved each epoch. To enable
            resumption of training after a crash, the user should initialise
            this callback with a :py:class:`SeqToSeq` instance.
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        load_fp: Optional[Union[str, Path]] = None,
        *args,
        **kwargs,
    ):
        self.log_dir = Path(log_dir)
        self.load_fp = load_fp
        super().__init__(*args, **kwargs)

    def on_train_begin(self, **kwargs) -> Optional[Dict]:
        if self.load_fp is not None:
            epoch, total_train_batches = self._load_seq_to_seq(self.load_fp)
        else:
            epoch, total_train_batches = self._load_most_recent_state_dict()

        # Update CallbackHandler state_dict
        if epoch is not None:
            kwargs["epoch"] = epoch + 1
        if total_train_batches is not None:
            kwargs["total_train_batches"] = total_train_batches

        if kwargs["epoch"] > kwargs["epochs"]:
            warnings.warn(
                f'cb_handler.state_dict["epoch"] is > '
                f'cb_handler.state_dict["epochs"] so no training will occur.'
            )
        return kwargs

    def _load_most_recent_state_dict(
        self,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Loads most recent state_dict in ``self.log_dir``.

        ''Most recent'' in this case is chosen according to state filenames
        which must be in the form 'state_dict_<EPOCH>.pt'.

        Returns:
            A Tuple of Optional ints ``epoch, total_train_batches``.
        """
        fnames = []
        dict_fname = None
        for fname in os.listdir(self.log_dir):
            if re.match(r"state_dict_\d*\.pt", fname):
                epochs = [int(x) for x in re.findall(r"\d*", fname) if x != ""]
                fnames.append((epochs[0], fname))

        epoch: Optional[int] = None
        total_train_batches: Optional[int] = None
        if len(fnames) == 0:
            return epoch, total_train_batches

        fnames.sort()
        fname_epoch, dict_fname = fnames[-1]
        dict_fpath = self.log_dir / dict_fname
        epoch, total_train_batches = self._load_seq_to_seq(dict_fpath)

        if fname_epoch != epoch:
            warnings.warn(
                "Saved state_dict epoch did not match the epoch"
                f"parsed from the filename. Loaded state from "
                f"{str(dict_fpath)} but this might not be the most "
                f"recent state_dict in ``log_dir``."
            )
        return epoch, total_train_batches

    def _load_seq_to_seq(
        self, state_dict_fp: Union[str, Path]
    ) -> Tuple[Optional[int], Optional[int]]:
        """Loads ``seq_to_seq`` state dict from path.

        Args:
            state_dict_fp: A path to a state dict.

        Returns:
            A Tuple of Optional ints where the first element is the number of
            ``epoch``s completed and the second is the ``total_train_batches``
            seen.
        """
        dict_ = torch.load(state_dict_fp)
        epoch = dict_.pop("epoch", None)
        total_train_batches = dict_.pop("total_train_batches", None)
        self.model.load_state_dict(dict_)

        return epoch, total_train_batches

    def on_epoch_end(self, **kwargs):
        if not self.training:
            return
        dict_ = self.model.state_dict()
        dict_["epoch"] = kwargs["epoch"]
        dict_["total_train_batches"] = kwargs["total_train_batches"]
        torch.save(
            dict_,
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
