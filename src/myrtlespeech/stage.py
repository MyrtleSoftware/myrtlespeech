import enum

from myrtlespeech.protos import stage_pb2


class Stage(enum.Enum):
    """TODO"""

    TRAIN = stage_pb2.Stage.TRAIN
    EVAL = stage_pb2.Stage.EVAL
    TRAIN_AND_EVAL = stage_pb2.Stage.TRAIN_AND_EVAL
