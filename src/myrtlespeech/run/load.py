from pathlib import Path
from typing import Dict
from typing import Union

import torch
from myrtlespeech.model.seq_to_seq import SeqToSeq


def load_seq_to_seq(
    seq_to_seq: SeqToSeq, state_dict_fp: Union[str, Path]
) -> Dict:
    """Loads ``seq_to_seq`` state dict from path and returns training state.

    .. note::

        The correct training state will only be returned if the
        saved state dict was created with the :py:class:`Saver`
        callback.

    Args:
        seq_to_seq: A :py:class:`.SeqToSeq` model.

        state_dict_fp: A path to a ``seq_to_seq`` state dict.

    Returns:
        A Dict containing the training state of the loaded state_dict
        :py:class:`.SeqToSeq` if known. This state includes the number of
        epochs completed and the total number of batches seen and should be
        passed as ``training_state`` to :py:func:`fit` in order to resume
        training from the same point.
    """
    dict_ = torch.load(state_dict_fp)
    epoch = dict_.pop("epoch", None)
    total_train_batches = dict_.pop("total_train_batches", None)
    seq_to_seq.load_state_dict(dict_)

    training_state = {}
    if epoch is None:
        # attempt to parse epoch from filename
        fname = Path(state_dict_fp).name
        epoch_str = fname.replace("state_dict_", "").replace(".pt", "")
        if epoch_str.isnumeric():
            epoch = int(epoch_str)
    if epoch is not None:
        training_state["epoch"] = epoch
    if total_train_batches is not None:
        training_state["total_train_batches"] = total_train_batches

    return training_state
