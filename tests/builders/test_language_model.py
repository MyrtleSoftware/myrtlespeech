from typing import Callable
from typing import Optional
from typing import Tuple

import pytest
from hypothesis import given
from myrtlespeech.builders.language_model import build
from myrtlespeech.protos import language_model_pb2

from tests.protos.test_language_model import language_models


# Utilities -------------------------------------------------------------------


def language_model_module_match_cfg(
    lm: Optional[Callable[[Tuple[int, ...]], float]],
    lm_cfg: language_model_pb2.LanguageModel,
) -> None:
    """Ensures language model matches protobuf configuration."""
    supported_lm = lm_cfg.WhichOneof("supported_lms")

    if supported_lm == "no_lm":
        assert lm is None
    else:
        raise ValueError(f"unknown language model {supported_lm}")


# Tests -----------------------------------------------------------------------


@given(lm_cfg=language_models())
def test_build_returns_correct_language_model_with_valid_params(
    lm_cfg: language_model_pb2.LanguageModel,
) -> None:
    """Test that build returns the correct language model with valid params."""
    lm = build(lm_cfg)
    language_model_module_match_cfg(lm, lm_cfg)


@given(lm_cfg=language_models())
def test_unknown_language_model_raises_value_error(
    lm_cfg: language_model_pb2.LanguageModel,
) -> None:
    """Ensures ValueError is raised when language model not supported.

    This can occur when the protobuf is updated and build is not.
    """
    lm_cfg.ClearField(lm_cfg.WhichOneof("supported_lms"))
    with pytest.raises(ValueError):
        build(lm_cfg)
