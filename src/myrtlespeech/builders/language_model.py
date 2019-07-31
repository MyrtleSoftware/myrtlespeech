from typing import Callable
from typing import Optional
from typing import Tuple

from myrtlespeech.protos import language_model_pb2


def build(
    lm_cfg: language_model_pb2.LanguageModel
) -> Optional[Callable[[Tuple[int, ...]], float]]:
    """Returns a language model based on the config.

    Args:
        lm_cfg: A ``LanguageModel`` protobuf object containing the config for
            the desired language model.

    Returns:
        A language model based on the config or None if ``no_lm``.

    Example:

        >>> from google.protobuf import text_format
        >>> lm_cfg_text = '''
        ... no_lm {
        ... }
        ... '''
        >>> lm_cfg = text_format.Merge(
        ...     lm_cfg_text,
        ...     language_model_pb2.LanguageModel()
        ... )
        >>> build(lm_cfg) is None
        True
    """
    supported_lm = lm_cfg.WhichOneof("supported_lms")
    if supported_lm == "no_lm":
        return None
    raise ValueError(f"{supported_lm} not supported")
