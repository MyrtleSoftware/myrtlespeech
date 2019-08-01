from typing import Tuple

import torch
from myrtlespeech.builders.dataset import build as build_dataset
from myrtlespeech.builders.speech_to_text import build as build_stt
from myrtlespeech.data.batch import seq_to_seq_collate_fn
from myrtlespeech.model.seq_to_seq import SeqToSeq
from myrtlespeech.protos import task_config_pb2


def build(
    task_config: task_config_pb2.TaskConfig, seq_len_support: bool = True
) -> Tuple[
    SeqToSeq,
    int,
    torch.optim.Optimizer,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
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
        A tuple of ``(model, epochs, optim, train_loader, eval_loader)`` where:

            model:
                A :py:class:`.SeqToSeq` model.

            epochs:
                The number of epochs to train for.

            optim:
                A :py:class:`torch.optim.Optimizer` initialised with the
                ``model`` parameters.

            train_loader:
                A :py:class:`torch.utils.data.DataLoader` for the training data.

            eval_loader:
                A :py:class:`torch.utils.data.DataLoader` for the eval data.

    Raises:
        :py:class:`ValueError`: On invalid configuration.
    """
    model_str = task_config.WhichOneof("supported_models")
    if model_str == "speech_to_text":
        model = build_stt(
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
        transform=model.pre_process,
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
        transform=model.pre_process,
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
