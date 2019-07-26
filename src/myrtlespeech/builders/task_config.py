from typing import Dict, List, Tuple

import torch

from myrtlespeech.builders.dataset import build as build_dataset
from myrtlespeech.builders.speech_to_text import build as build_stt
from myrtlespeech.protos import task_config_pb2


def pad_sequence(
    sequences: List[torch.Tensor], padding_value: int = 0
) -> torch.Tensor:
    """TODO: refactor/document/test

    Args:
        sequences: TODO: List of sequences with size ``[*, L]``. Assume all
        sequences have same ``*`` sizes.

    Returns:
        TODO: `[B, *, L]`
    """
    max_size = sequences[0].size()
    leading_dims = max_size[:-1]
    max_len = max([s.size(-1) for s in sequences])

    out_dims = (len(sequences),) + leading_dims + (max_len,)

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(-1)
        out_tensor[i, ..., :length] = tensor

    return out_tensor


def collate_fn(
    batch: List[
        Tuple[
            Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
        ]
    ]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """TODO: refactor/document/test/move to appropriate place"""
    inputs, in_seq_lens = [], []
    targets, target_seq_lens = [], []

    for (input, in_seq_len), (target, target_seq_len) in batch:
        inputs.append(input)
        in_seq_lens.append(in_seq_len)
        targets.append(target)
        target_seq_lens.append(target_seq_len)

    inputs = pad_sequence(inputs)
    in_seq_lens = torch.tensor(in_seq_lens, requires_grad=False)
    targets = pad_sequence(targets)
    target_seq_lens = torch.tensor(target_seq_lens, requires_grad=False)

    xb = {"x": inputs, "seq_lens": in_seq_lens}
    yb = {"targets": targets, "target_lengths": target_seq_lens}

    return xb, yb


def build(
    task_config: task_config_pb2.TaskConfig, seq_len_support: bool = True
) -> Tuple[
    torch.nn.Module, int, torch.optim.Optimizer, torch.utils.data.DataLoader
]:
    """TODO


    TODO: Where to put target_transform? Currently it's shoe-horned in here.


    """
    model = build_stt(task_config.model, seq_len_support=seq_len_support)

    # create optimizer
    optim_str = task_config.train_config.WhichOneof("supported_optimizers")
    if optim_str == "sgd":
        kwargs = {}

        sgd = task_config.train_config.sgd
        if sgd.HasField("momentum"):
            kwargs["momentum"] = sgd.momentum.value

        if sgd.HasField("l2_weight_decay"):
            kwargs["weight_decay"] = sgd.l2_weight_decay.value

        optim = torch.optim.SGD(
            params=model.parameters(), lr=sgd.learning_rate, **kwargs
        )
    else:
        raise ValueError("unsupported optimizer {optim_str}")

    # create dataloader
    def target_transform(target):
        return torch.tensor(
            model.alphabet.get_indices(target), requires_grad=False
        )

    dataset = build_dataset(
        task_config.train_config.dataset,
        transform=model.transform,
        target_transform=target_transform,
        add_seq_len_to_transforms=seq_len_support,
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=task_config.train_config.batch_size,
        collate_fn=collate_fn,
    )

    return model, task_config.train_config.epochs, optim, loader
