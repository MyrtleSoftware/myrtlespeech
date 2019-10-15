from myrtlespeech.model.lookahead import Lookahead
from myrtlespeech.protos import lookahead_pb2


def build(
    lookahead_cfg: lookahead_pb2.Lookahead, input_features: int
) -> Lookahead:
    """Returns a :py:class:`.Lookahead` based on the config.

    Args:
        lookahead_cfg: A ``Lookahead`` protobuf object containing the config
        for the desired :py:class:`.Lookahead`.

        input_features: The number of features for the input.

    Returns:
        A :py:class:`Lookahead` based on the config.

    Example:
        >>> from google.protobuf import text_format
        >>> cfg_text = '''
        ... context: 80;
        ... '''
        >>> cfg = text_format.Merge(
        ...     cfg_text,
        ...     lookahead_pb2.Lookahead()
        ... )
        >>> build(cfg, input_features=32)
        Lookahead(in_features=32, context=80)
    """
    return Lookahead(in_features=input_features, context=lookahead_cfg.context)
