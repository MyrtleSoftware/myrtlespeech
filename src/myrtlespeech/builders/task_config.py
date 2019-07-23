from typing import Tuple

import torch

from myrtlespeech.builders.dataset import build as build_dataset
from myrtlespeech.builders.speech_to_text import build as build_stt
from myrtlespeech.protos import task_config_pb2


def build(
    task_config: task_config_pb2.TaskConfig, seq_len_support: bool = False
) -> Tuple[
    torch.nn.Module, int, torch.optim.Optimizer, torch.utils.data.DataLoader
]:
    """TODO


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
    dataset = build_dataset(
        task_config.train_config.dataset,
        transform=model.get_transform(),
        target_transform=None,
        add_seq_len_to_transforms=seq_len_support,
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=task_config.train_config.batch_size
    )

    return model, task_config.train_config.epochs, optim, loader
