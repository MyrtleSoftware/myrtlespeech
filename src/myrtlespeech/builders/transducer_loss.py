from myrtlespeech.loss.transducer_loss import TransducerLoss
from myrtlespeech.protos import transducer_loss_pb2


def build(
    transducer_loss_cfg: transducer_loss_pb2.TransducerLoss
) -> TransducerLoss:
    r"""Returns a :py:class:`.TransducerLoss` based on the config.

    Args:
        transducer_loss_cfg: A ``TransducerLoss`` protobuf object containing
            the config for the desired :py:class:`.TransducerLoss`.

    Returns:
        A :py:class:`.TransducerLoss` based on the config.

    Example:
        >>> from google.protobuf import text_format
        >>> transducer_loss_cfg_text = '''
        ... blank_index: 0;
        ... reduction: SUM;
        ... '''
        >>> transducer_loss_cfg = text_format.Merge(
        ...     transducer_loss_cfg_text,
        ...     transducer_loss_pb2.TransducerLoss()
        ... )
        >>> build(transducer_loss_cfg)
        TransducerLoss(
          (transducer_loss): RNNTLoss()
        )

    """
    reduction_map = {
        transducer_loss_pb2.TransducerLoss.NONE: "none",
        transducer_loss_pb2.TransducerLoss.MEAN: "mean",
        transducer_loss_pb2.TransducerLoss.SUM: "sum",
    }
    try:
        reduction = reduction_map[transducer_loss_cfg.reduction]
    except KeyError:
        raise ValueError(
            f"reduction={transducer_loss_cfg.reduction} not supported"
        )

    transducer_loss = TransducerLoss(
        blank=transducer_loss_cfg.blank_index, reduction=reduction
    )

    return transducer_loss
