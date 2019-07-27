from typing import Tuple

import torch

from myrtlespeech.builders.dataset import build as build_dataset
from myrtlespeech.builders.speech_to_text import build as build_stt
from myrtlespeech.data.batch import collate_fn
from myrtlespeech.protos import task_config_pb2


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
