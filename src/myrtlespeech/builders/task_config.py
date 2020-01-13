import multiprocessing
from typing import Tuple

import torch
from myrtlespeech.builders.dataset import build as build_dataset
from myrtlespeech.builders.lr_scheduler import build as build_lr_scheduler
from myrtlespeech.builders.speech_to_text import build as build_stt
from myrtlespeech.data.batch import seq_to_seq_collate_fn
from myrtlespeech.data.sampler import RandomBatchSampler
from myrtlespeech.model.seq_to_seq import SeqToSeq
from myrtlespeech.protos import task_config_pb2
from myrtlespeech.run.callbacks.accumulation import GradientAccumulation


def build(
    task_config: task_config_pb2.TaskConfig, accumulation_steps: int = 1,
) -> Tuple[
    SeqToSeq, int, torch.utils.data.DataLoader, torch.utils.data.DataLoader
]:
    """Builds a ``task_config`` and returns each component.

    Args:
        task_config: A :py:class:`task_config_pb2.TaskConfig` protobuf object
            containing the config for the desired task.

        accumulation_steps: Number of steps to perform gradient accumulation
            over.

    Returns:
        A tuple of ``(seq_to_seq, epochs, optim, train_loader, eval_loader)``
        where:

            seq_to_seq:
                A :py:class:`.SeqToSeq` model.

            epochs:
                The number of epochs to train for.

            train_loader:
                A :py:class:`torch.utils.data.DataLoader` for the training
                data.

            eval_loader:
                A :py:class:`torch.utils.data.DataLoader` for the eval data.

    Raises:
        :py:class:`ValueError`: On invalid configuration.
    """
    model_str = task_config.WhichOneof("supported_models")
    if model_str == "speech_to_text":
        seq_to_seq = build_stt(task_config.speech_to_text)
    else:
        raise ValueError(f"unsupported model {model_str}")

    train_batch_size = task_config.train_config.batch_size
    if train_batch_size % accumulation_steps != 0:
        raise ValueError(
            f"train_batch_size must be divisible by ",
            f"accumulation_steps but train_batch_size="
            f"{train_batch_size} and accumulation_steps="
            f"{accumulation_steps}.",
        )
    train_batch_size = train_batch_size // accumulation_steps
    if train_batch_size == 0:
        raise ValueError(
            "task_config.train_config.batch_size "
            "// accumulation_steps == 0."
        )
    # create optimizer
    optim_str = task_config.train_config.WhichOneof("supported_optimizers")
    if optim_str == "sgd":
        kwargs = {}

        sgd = task_config.train_config.sgd
        if sgd.HasField("momentum"):
            kwargs["momentum"] = sgd.momentum.value

        if sgd.HasField("l2_weight_decay"):
            kwargs["weight_decay"] = sgd.l2_weight_decay.value

        kwargs["nesterov"] = sgd.nesterov_momentum

        optim = torch.optim.SGD(
            params=seq_to_seq.parameters(), lr=sgd.learning_rate, **kwargs
        )
    elif optim_str == "adam":
        kwargs = {}

        adam = task_config.train_config.adam
        betas = [0.9, 0.999]
        if adam.HasField("beta_1"):
            betas[0] = adam.beta_1.value
        if adam.HasField("beta_2"):
            betas[1] = adam.beta_2.value
        betas = tuple(betas)  # type: ignore

        if adam.HasField("eps"):
            kwargs["eps"] = adam.eps.value

        if adam.HasField("l2_weight_decay"):
            kwargs["weight_decay"] = adam.l2_weight_decay.value

        kwargs["amsgrad"] = adam.amsgrad

        optim = torch.optim.Adam(
            params=seq_to_seq.parameters(), lr=adam.learning_rate, **kwargs
        )
    else:
        raise ValueError(f"unsupported optimizer {optim_str}")

    seq_to_seq.optim = optim

    # create dataloader
    def target_transform(target):
        return torch.tensor(
            seq_to_seq.alphabet.get_indices(target),
            dtype=torch.int32,
            requires_grad=False,
        )

    num_workers = multiprocessing.cpu_count() // 4

    # training
    train_dataset = build_dataset(
        task_config.train_config.dataset,
        transform=seq_to_seq.pre_process,
        target_transform=target_transform,
        add_seq_len_to_transforms=True,
    )

    shuffle = task_config.train_config.shuffle_batches_before_every_epoch
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=RandomBatchSampler(
            indices=range(len(train_dataset)),
            batch_size=train_batch_size,
            shuffle=shuffle,
            drop_last=False,
        ),
        num_workers=num_workers,
        collate_fn=seq_to_seq_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    # Add accumulation callback
    train_loader.callbacks = [
        GradientAccumulation(accumulation_steps=accumulation_steps)
    ]

    # eval
    eval_dataset = build_dataset(
        task_config.eval_config.dataset,
        transform=seq_to_seq.pre_process,
        target_transform=target_transform,
        add_seq_len_to_transforms=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=task_config.eval_config.batch_size,
        num_workers=num_workers,
        collate_fn=seq_to_seq_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    # create learning rate scheduler
    seq_to_seq.lr_scheduler = build_lr_scheduler(
        train_config=task_config.train_config,
        optimizer=seq_to_seq.optim,
        batches_per_epoch=len(train_loader) // accumulation_steps,
        epochs=task_config.train_config.epochs,
    )

    return (
        seq_to_seq,
        task_config.train_config.epochs,
        train_loader,
        eval_loader,
    )
