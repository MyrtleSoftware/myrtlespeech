from typing import Tuple

from myrtlespeech.data.preprocess import MFCC
from myrtlespeech.protos import pre_process_step_pb2


def build(
    pre_process_step_cfg: pre_process_step_pb2.PreProcessStep,
) -> Tuple[MFCC, bool]:
    """Returns tuple of (preprocessing callable, bool indicating if train only).

    Args:
        pre_process_step_cfg: A ``PreProcessStep`` protobuf object containing
            the config for the desired preprocessing step.

    Returns:
        A tuple of (preprocessing callable, bool indicating if train only).

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
    else:
        raise ValueError("unknown pre_process_step '{step_type}'")

    return (step, pre_process_step_cfg.train_only)
