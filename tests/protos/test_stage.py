import hypothesis.strategies as st
from myrtlespeech.protos import stage_pb2


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def stages(draw) -> st.SearchStrategy:
    """Returns a SearchStrategy for a stage."""
    return draw(st.sampled_from(stage_pb2.Stage.values()))
