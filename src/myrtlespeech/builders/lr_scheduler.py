import torch
from myrtlespeech.lr_scheduler.base import LRSchedulerBase
from myrtlespeech.lr_scheduler.constant import ConstantLR
from myrtlespeech.lr_scheduler.poly import PolynomialLR
from myrtlespeech.protos import train_config_pb2


def build(
    train_config: train_config_pb2.TrainConfig,
    optimizer: torch.optim.Optimizer,
    batches_per_epoch: int,
    epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Builds a ``learning rate scheduler`` and returns it.

    Args:
        train_config: A :py:class:`train_config_pb2.TrainConfig` protobuf
            object containing the config for the desired training task.

        optimizer: A :py:class:`torch.optim` that is used in the learning rate
            scheduler initialization.

        batches_per_epoch: Number of batches in a single epoch.

        epochs: Total number of epochs.

    Returns:
        A :py:class:`torch.optim.lr_scheduler._LRScheduler` instance.

    Raises:
        :py:class:`ValueError`: On invalid configuration.
    """

    lr_scheduler_str = train_config.WhichOneof("supported_lr_scheduler")

    if lr_scheduler_str == "constant_lr":
        lr_scheduler = ConstantLR(optimizer=optimizer)
    elif lr_scheduler_str == "step_lr":
        kwargs = {}

        step_lr = train_config.step_lr
        if step_lr.HasField("gamma"):
            kwargs["gamma"] = step_lr.gamma.value
        kwargs["step_size"] = step_lr.step_size

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, **kwargs
        )
    elif lr_scheduler_str == "exponential_lr":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=train_config.exponential_lr.gamma,
        )
    elif lr_scheduler_str == "cosine_annealing_lr":
        kwargs = {}

        cosine_annealing_lr = train_config.cosine_annealing_lr
        if cosine_annealing_lr.HasField("eta_min"):
            kwargs["eta_min"] = cosine_annealing_lr.eta_min.value
        kwargs["T_max"] = cosine_annealing_lr.t_max

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, **kwargs
        )
    elif lr_scheduler_str == "polynomial_lr":
        if epochs < 1:
            raise ValueError(f"Cannot have < 1 epochs for polynomial_lr.")
        total_steps = batches_per_epoch * epochs
        lr_scheduler = PolynomialLR(
            optimizer=optimizer, total_steps=total_steps
        )

    else:
        raise ValueError(
            f"unsupported learning rate scheduler {lr_scheduler_str}"
        )

    # get scheduler step frequency
    if lr_scheduler_str in [
        "step_lr",
        "exponential_lr",
        "cosine_annealing_lr",
    ]:
        step_freq = batches_per_epoch  # Step at end of each epoch
    elif lr_scheduler_str in ["polynomial_lr"]:
        step_freq = 1  # Step after every batch
    elif lr_scheduler_str == "constant_lr":
        step_freq = batches_per_epoch * 100  # Arbitrary large value

    # Add lr warmup iff num_warmup_steps is not None
    num_warmup_steps = None
    if train_config.HasField("lr_warmup"):
        num_warmup_steps = train_config.lr_warmup.num_warmup_steps

    return LRSchedulerBase(
        scheduler=lr_scheduler,
        scheduler_step_freq=step_freq,
        num_warmup_steps=num_warmup_steps,
    )
