from typing import Tuple
from typing import Union

from myrtlespeech.data.preprocess import MFCC
from myrtlespeech.data.preprocess import Standardize
from myrtlespeech.protos import pre_process_step_pb2
from myrtlespeech.run.stage import Stage


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
            numcep=pre_process_step_cfg.mfcc.numcep,
            winlen=pre_process_step_cfg.mfcc.winlen,
            winstep=pre_process_step_cfg.mfcc.winstep,
        )
    elif step_type == "standardize":
        """TODO: tidy"""
        step = Standardize(
            mean=pre_process_step_cfg.standardize.mean,
            dim=pre_process_step_cfg.standardize.reduce_dim,
        )
    else:
        raise ValueError("unknown pre_process_step '{step_type}'")

    return step, Stage(pre_process_step_cfg.stage)
