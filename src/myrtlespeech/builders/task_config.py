import multiprocessing
from typing import Callable
from typing import Tuple

import torch
from myrtlespeech.builders.dataset import build as build_dataset
from myrtlespeech.builders.speech_to_text import build as build_stt
from myrtlespeech.data.alphabet import Alphabet
from myrtlespeech.data.batch import seq_to_seq_collate_fn
from myrtlespeech.data.sampler import RandomBatchSampler
from myrtlespeech.model.seq_to_seq import SeqToSeq
from myrtlespeech.protos import task_config_pb2
from myrtlespeech.protos import train_config_pb2


def build(
    task_config: task_config_pb2.TaskConfig,
) -> Tuple[
    SeqToSeq, int, torch.utils.data.DataLoader, torch.utils.data.DataLoader
]:
    """Builds a ``task_config`` and returns each component.

    Args:
        task_config: A :py:class:`task_config_pb2.TaskConfig` protobuf object
            containing the config for the desired task.

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

    # create learning rate scheduler
    seq_to_seq.lr_scheduler = _create_lr_scheduler(
        task_config, seq_to_seq.optim
    )

    # get target transforms
    train_target_trans, eval_target_trans = _get_target_transform(
        task_config.train_config, seq_to_seq.alphabet
    )

    num_workers = multiprocessing.cpu_count() // 4

    # training
    train_dataset = build_dataset(
        task_config.train_config.dataset,
        transform=seq_to_seq.pre_process,
        target_transform=train_target_trans,
        add_seq_len_to_transforms=True,
    )

    shuffle = task_config.train_config.shuffle_batches_before_every_epoch
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=RandomBatchSampler(
            indices=range(len(train_dataset)),
            batch_size=task_config.train_config.batch_size,
            shuffle=shuffle,
            drop_last=False,
        ),
        num_workers=num_workers,
        collate_fn=seq_to_seq_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    # eval
    eval_dataset = build_dataset(
        task_config.eval_config.dataset,
        transform=seq_to_seq.pre_process,
        target_transform=eval_target_trans,
        add_seq_len_to_transforms=True,
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


def _create_lr_scheduler(
    task_config: task_config_pb2.TaskConfig, optimizer: torch.optim
) -> torch.optim.lr_scheduler:
    """Builds a ``learning rate scheduler`` and returns it.

    Args:
        task_config: A :py:class:`task_config_pb2.TaskConfig` protobuf object
            containing the config for the desired task.

        optimizer: A :py:class:`torch.optim` that is used in the learning rate
            scheduler initialization.

    Returns:
        A :py:class:`torch.optim.lr_scheduler` instance.

    Raises:
        :py:class:`ValueError`: On invalid configuration.
        """
    lr_scheduler_str = task_config.train_config.WhichOneof(
        "supported_lr_scheduler"
    )

    if lr_scheduler_str == "constant_lr":
        lr_scheduler = None
    elif lr_scheduler_str == "step_lr":
        kwargs = {}

        step_lr = task_config.train_config.step_lr
        if step_lr.HasField("gamma"):
            kwargs["gamma"] = step_lr.gamma.value
        kwargs["step_size"] = step_lr.step_size

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, **kwargs
        )
    elif lr_scheduler_str == "exponential_lr":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=task_config.train_config.exponential_lr.gamma,
        )
    elif lr_scheduler_str == "cosine_annealing_lr":
        kwargs = {}

        cosine_annealing_lr = task_config.train_config.cosine_annealing_lr
        if cosine_annealing_lr.HasField("eta_min"):
            kwargs["eta_min"] = cosine_annealing_lr.eta_min.value
        kwargs["T_max"] = cosine_annealing_lr.t_max

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, **kwargs
        )
    else:
        raise ValueError(
            f"unsupported learning rate scheduler {lr_scheduler_str}"
        )

    return lr_scheduler


def _get_target_transform(
    train_config: train_config_pb2.TrainConfig, alphabet: Alphabet
) -> Tuple[Callable, Callable]:
    """TODO"""

    def target_transform(target):
        return torch.tensor(
            alphabet.get_indices(target),
            dtype=torch.int32,
            requires_grad=False,
        )

    vocab_size = len(alphabet)
    transform = target_transform
    if train_config.HasField("label_smoothing"):
        type_idx = train_config.label_smoothing.type
        smoothing_idx_to_name = (
            train_config_pb2.TrainConfig.LabelSmoothing.SmoothingType.Name
        )
        type_str = smoothing_idx_to_name(type_idx).lower()
        probability = train_config.label_smoothing.probability.value
        assert 0.0 < probability < 1
        if type_str == "uniform":

            def uniform(target):
                target = target_transform(target)
                uniform = torch.rand(target.shape)
                mask = probability > uniform
                if mask.sum() > 0:
                    new_values = (
                        torch.LongTensor(mask.sum().item())
                        .random_(0, vocab_size - 1)
                        .to(target.dtype)
                    )
                    target.masked_scatter_(mask, new_values)
                return target

            transform = uniform

        elif type_str == "unigram":
            raise NotImplementedError
        else:
            raise ValueError(
                f"label_smoothing_type={type_str} is not recognized."
            )

    return transform, target_transform
