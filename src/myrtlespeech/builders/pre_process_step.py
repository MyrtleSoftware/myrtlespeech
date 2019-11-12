from typing import Tuple
from typing import Union

import torch
from myrtlespeech.data.preprocess import AddContextFrames
from myrtlespeech.data.preprocess import Standardize
from myrtlespeech.protos import pre_process_step_pb2
from myrtlespeech.run.stage import Stage
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import MFCC


class LogMelFB:
    r"""Wrapper on `torchaudio.transforms.MelSpectrogram` that applies log.

    Args:
        See `torchaudio.transforms.MelSpectrogram`

    Returns:
        See `torchaudio.transforms.MelSpectrogram`
    """

    def __init__(self, **kwargs):
        self.mel_spectogram = MelSpectrogram(**kwargs)

    def __call__(self, waveform):
        r"""See initization docstring."""
        feat = self.mel_spectogram(waveform)

        # Numerical stability:
        feat = torch.where(
            feat == 0, torch.tensor(torch.finfo(waveform.dtype).eps), feat
        )

        return feat.log()


def build(
    pre_process_step_cfg: pre_process_step_pb2.PreProcessStep,
) -> Tuple[Union[MFCC, Standardize, AddContextFrames, LogMelFB], Stage]:
    """Returns tuple of ``(preprocessing callable, stage)``.

    Args:
        pre_process_step_cfg: A ``PreProcessStep`` protobuf object containing
            the config for the desired preprocessing step.

    Returns:
        A tuple of ``(preprocessing callable, stage)``.

    Raises:
        :py:class:`ValueError`: On invalid configuration.
    """
    step_type = pre_process_step_cfg.WhichOneof("pre_process_step")
    if step_type == "mfcc":
        step = MFCC(
            n_mfcc=pre_process_step_cfg.mfcc.n_mfcc,
            melkwargs={
                "win_length": pre_process_step_cfg.mfcc.win_length,
                "hop_length": pre_process_step_cfg.mfcc.hop_length,
            },
        )
    elif step_type == "lmfb":
        step = LogMelFB(
            n_mels=pre_process_step_cfg.lmfb.n_mels,
            win_length=pre_process_step_cfg.lmfb.win_length,
            hop_length=pre_process_step_cfg.lmfb.hop_length,
        )
    elif step_type == "standardize":
        step = Standardize()
    elif step_type == "context_frames":
        step = AddContextFrames(
            n_context=pre_process_step_cfg.context_frames.n_context
        )
    else:
        raise ValueError(f"unknown pre_process_step '{step_type}'")

    return step, Stage(pre_process_step_cfg.stage)
