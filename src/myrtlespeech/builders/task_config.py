from typing import Tuple

import torch
from myrtlespeech.builders.dataset import build as build_dataset
from myrtlespeech.builders.seq_to_seq import build as build_s2s
from myrtlespeech.data.batch import seq_to_seq_collate_fn
from myrtlespeech.protos import task_config_pb2


def build(
    task_config: task_config_pb2.TaskConfig, seq_len_support: bool = True
) -> Tuple[
    torch.nn.Module,
    int,
    torch.optim.Optimizer,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """TODO

    """
    model = build_s2s(task_config.model, seq_len_support=seq_len_support)

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
        raise ValueError(f"unsupported optimizer {optim_str}")

    # create dataloader
    def target_transform(target):
        return torch.tensor(
            model.alphabet.get_indices(target),
            dtype=torch.int32,
            requires_grad=False,
        )

    # training
    train_dataset = build_dataset(
        task_config.train_config.dataset,
        transform=model.transform,
        target_transform=target_transform,
        add_seq_len_to_transforms=seq_len_support,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=task_config.train_config.batch_size,
        collate_fn=seq_to_seq_collate_fn,
        shuffle=False,
    )

    # eval
    eval_dataset = build_dataset(
        task_config.eval_config.dataset,
        transform=model.transform,
        target_transform=target_transform,
        add_seq_len_to_transforms=seq_len_support,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=task_config.eval_config.batch_size,
        collate_fn=seq_to_seq_collate_fn,
    )

    return (
        model,
        task_config.train_config.epochs,
        optim,
        train_loader,
        eval_loader,
    )
