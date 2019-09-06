import multiprocessing
from typing import Tuple

import torch
from myrtlespeech.builders.dataset import build as build_dataset
from myrtlespeech.builders.speech_to_text import build as build_stt
from myrtlespeech.data.batch import seq_to_seq_collate_fn
from myrtlespeech.data.sampler import RandomBatchSampler
from myrtlespeech.model.seq_to_seq import SeqToSeq
from myrtlespeech.protos import task_config_pb2


def build(
    task_config: task_config_pb2.TaskConfig, seq_len_support: bool = True
) -> Tuple[
    SeqToSeq, int, torch.utils.data.DataLoader, torch.utils.data.DataLoader
]:
    """Builds a ``task_config`` and returns each component.

    Args:
        task_config: A :py:class:`task_config_pb2.TaskConfig` protobuf object
            containing the config for the desired task.

        seq_len_support: If :py:data:`True`, the
            :py:meth:`torch.nn.Module.forward` method of the returned
            :py:data:`.SeqToSeq.model` must optionally accept a ``seq_lens``
            kwarg.

    Returns:
        A tuple of ``(seq_to_seq, epochs, optim, train_loader, eval_loader)`` where:

            seq_to_seq:
                A :py:class:`.SeqToSeq` model.

            epochs:
                The number of epochs to train for.

            train_loader:
                A :py:class:`torch.utils.data.DataLoader` for the training data.

            eval_loader:
                A :py:class:`torch.utils.data.DataLoader` for the eval data.

    Raises:
        :py:class:`ValueError`: On invalid configuration.
    """
    model_str = task_config.WhichOneof("supported_models")
    if model_str == "speech_to_text":
        seq_to_seq = build_stt(
            task_config.speech_to_text, seq_len_support=seq_len_support
        )
    else:
        raise ValueError(f"unsupported model {model_str}")

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
        add_seq_len_to_transforms=seq_len_support,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=RandomBatchSampler(
            indices=range(len(train_dataset)),
            batch_size=task_config.train_config.batch_size,
            shuffle=task_config.train_config.shuffle_batches_before_every_epoch,
            drop_last=False
        ),
        num_workers=num_workers,
        collate_fn=seq_to_seq_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    # eval
    eval_dataset = build_dataset(
        task_config.eval_config.dataset,
        transform=seq_to_seq.pre_process,
        target_transform=target_transform,
        add_seq_len_to_transforms=seq_len_support,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=task_config.eval_config.batch_size,
        num_workers=num_workers,
        collate_fn=seq_to_seq_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        seq_to_seq,
        task_config.train_config.epochs,
        train_loader,
        eval_loader,
    )
