from typing import Tuple
from typing import Union

from myrtlespeech.data.preprocess import AddContextFrames
from myrtlespeech.data.preprocess import SpecAugment
from myrtlespeech.data.preprocess import Standardize
from myrtlespeech.protos import pre_process_step_pb2
from myrtlespeech.run.stage import Stage
from torchaudio.transforms import MFCC


def build(
    pre_process_step_cfg: pre_process_step_pb2.PreProcessStep,
) -> Tuple[Union[MFCC, Standardize], Stage]:
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
    elif step_type == "spec_augment":
        spec = pre_process_step_cfg.spec_augment
        step = SpecAugment(
            feature_mask=spec.feature_mask,
            time_mask=spec.time_mask,
            n_feature_masks=spec.n_feature_masks,
            n_time_masks=spec.n_time_masks
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
