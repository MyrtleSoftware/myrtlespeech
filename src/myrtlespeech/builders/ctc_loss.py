from myrtlespeech.loss.ctc_loss import CTCLoss
from myrtlespeech.protos import ctc_loss_pb2


def build(ctc_loss_cfg: ctc_loss_pb2.CTCLoss) -> CTCLoss:
    """Returns a :py:class:`.CTCLoss` based on the config.

    Args:
        ctc_loss_cfg: A ``CTCLoss`` protobuf object containing the config for
            the desired :py:class:`.CTCLoss`.

    Returns:
        A :py:class:`.CTCLoss` based on the config.

    Example:

        >>> from google.protobuf import text_format
        >>> ctc_loss_cfg_text = '''
        ... blank_index: 0;
        ... reduction: SUM;
        ... '''
        >>> ctc_loss_cfg = text_format.Merge(
        ...     ctc_loss_cfg_text,
        ...     ctc_loss_pb2.CTCLoss()
        ... )
        >>> build(ctc_loss_cfg)
        CTCLoss()
    """
    reduction_map = {
        ctc_loss_pb2.CTCLoss.NONE: "none",
        ctc_loss_pb2.CTCLoss.MEAN: "mean",
        ctc_loss_pb2.CTCLoss.SUM: "sum",
    }
    try:
        reduction = reduction_map[ctc_loss_cfg.reduction]
    except KeyError:
        raise ValueError(f"reduction={ctc_loss_cfg.reduction} not supported")

    ctc_loss = CTCLoss(blank=ctc_loss_cfg.blank_index, reduction=reduction)

    return ctc_loss
