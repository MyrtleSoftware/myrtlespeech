from myrtlespeech.loss.rnn_t_loss import RNNTLoss
from myrtlespeech.protos import rnn_t_loss_pb2


def build(rnn_t_loss_cfg: rnn_t_loss_pb2.RNNTLoss) -> RNNTLoss:
    """Returns a :py:class:`.RNNTLoss` based on the config.

    Args:
        rnn_t_loss_cfg: A ``RNNTLoss`` protobuf object containing the config for
            the desired :py:class:`.RNNTLoss`.

    Returns:
        A :py:class:`.RNNTLoss` based on the config.

    Example:
        >>> from google.protobuf import text_format
        >>> rnn_t_loss_cfg_text = '''
        ... blank_index: 0;
        ... reduction: SUM;
        ... '''
        >>> rnn_t_loss_cfg = text_format.Merge(
        ...     rnn_t_loss_cfg_text,
        ...     rnn_t_loss_pb2.RNNTLoss()
        ... )
        >>> build(rnn_t_loss_cfg)
        RNNTLoss(
          (rnnt_loss): RNNTLoss()
        )
    """
    reduction_map = {
        rnn_t_loss_pb2.RNNTLoss.NONE: "none",
        rnn_t_loss_pb2.RNNTLoss.MEAN: "mean",
        rnn_t_loss_pb2.RNNTLoss.SUM: "sum",
    }
    try:
        reduction = reduction_map[rnn_t_loss_cfg.reduction]
    except KeyError:
        raise ValueError(f"reduction={rnn_t_loss_cfg.reduction} not supported")

    rnn_t_loss = RNNTLoss(blank=rnn_t_loss_cfg.blank_index, reduction=reduction)

    return rnn_t_loss
