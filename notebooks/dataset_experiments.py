import os
import pathlib
import typing

import torch
from myrtlespeech.builders.task_config import build
from myrtlespeech.model.deep_speech_1 import DeepSpeech1
from myrtlespeech.post_process.utils import levenshtein
from myrtlespeech.protos import task_config_pb2
from myrtlespeech.run.callbacks.callback import Callback
from myrtlespeech.run.callbacks.callback import ModelCallback
from myrtlespeech.run.callbacks.clip_grad_norm import ClipGradNorm
from myrtlespeech.run.callbacks.csv_logger import CSVLogger
from myrtlespeech.run.callbacks.mixed_precision import MixedPrecision
from myrtlespeech.run.callbacks.report_mean_batch_loss import (
    ReportMeanBatchLoss,
)
from myrtlespeech.run.callbacks.stop_epoch_after import StopEpochAfter
from myrtlespeech.run.run import ReportCTCDecoder
from myrtlespeech.run.run import Saver
from myrtlespeech.run.run import TensorBoardLogger
from myrtlespeech.run.run import WordSegmentor
from myrtlespeech.run.stage import Stage
from myrtlespeech.run.train import fit


def f():
    # Import in function so mypy doesn't complain :(
    from google.protobuf import text_format

    with open("../src/myrtlespeech/configs/deep_speech_1_en.config") as f:
        return text_format.Merge(f.read(), task_config_pb2.TaskConfig())


task_config = f()

seq_to_seq, epochs, train_loader, eval_loader = build(task_config)

log_dir = "/home/samg/logs/ds1"
# train the model
fit(
    seq_to_seq,
    1000,  # epochs,
    train_loader=train_loader,
    eval_loader=eval_loader,
    callbacks=[
        # prof,
        ReportMeanBatchLoss(),
        ReportCTCDecoder(
            seq_to_seq.post_process, seq_to_seq.alphabet, WordSegmentor(" "),
        ),
        TensorBoardLogger(log_dir, seq_to_seq.model, histograms=False),
        MixedPrecision(seq_to_seq, opt_level="O1"),
        # ClipGradNorm(seq_to_seq, max_norm=400),
        # StopEpochAfter(epoch_batches=1),
        CSVLogger(
            f"{log_dir}/log.csv",
            exclude=[
                "epochs",
                # "reports/CTCGreedyDecoder/transcripts",
            ],
        ),
        Saver(log_dir, seq_to_seq),
    ],
)
