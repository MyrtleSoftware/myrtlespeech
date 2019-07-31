import warnings

from hypothesis import given

from myrtlespeech.builders.seq_to_seq import build
from myrtlespeech.model.seq_to_seq import SeqToSeq
from myrtlespeech.protos import seq_to_seq_pb2
from tests.protos.test_seq_to_seq import seq_to_seqs


# Tests -----------------------------------------------------------------------


@given(s2s_cfg=seq_to_seqs())
def test_build_returns_seq_to_seq(s2s_cfg: seq_to_seq_pb2.SeqToSeq) -> None:
    """Test that build returns a SeqToSeq instance."""
    assert isinstance(build(s2s_cfg), SeqToSeq)
    warnings.warn("SeqToSeq module only build and not checked if correct")
