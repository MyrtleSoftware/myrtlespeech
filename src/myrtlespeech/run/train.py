import torch


def fit(
    model: torch.nn.Module,
    epochs: int,
    optim: torch.optim.Optimizer,
    loader: torch.utils.data.DataLoader,
) -> None:
    """TODO


    """
    print(model, epochs, optim, loader)

    for epoch in range(epochs):
        for step, (input, target) in enumerate(loader):
            print(epoch, step)
            optim.zero_grad()

            output = model(input)

            loss = model.loss(output, target)

            loss.backward()

            optim.step()
