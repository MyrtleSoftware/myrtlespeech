import numpy as np
import torch
from myrtlespeech.loss.transducer_loss import TransducerLoss


def _get_log_probs_helper(eps):
    """Helper to return log probs.

    These probabilities are in the form described in
    `test_transducer_worked_example_1` docstring
    """
    probs_t1 = torch.tensor([[1.0, 0, 0], [0, 0, 1], [0, 0, 1]])
    probs_t2 = torch.tensor([[0.0, 0, 1], [0, 1, 0], [0, 0, 1]])
    probs_t3 = torch.tensor([[0, 0, 1.0], [0, 0, 1], [0, 0, 1]])
    probs = [probs_t1, probs_t2, probs_t3]
    probs = [x.unsqueeze(0) for x in probs]
    probs = torch.cat(probs, dim=0)
    probs = probs.unsqueeze(0).float()

    r = 2 * (1 - 3 * eps)
    probs *= r
    probs += 2 * eps
    probs /= 2.0
    log_probs = probs.log()
    log_lens = torch.IntTensor([3])  # T=3
    return log_probs, log_lens


def _label_collate_helper(labels):
    """Helper function that generates a tensor of targets.

    Generates a ``torch.Tensor`` from a List of torch.Tensors
    """

    batch_size = len(labels)
    target_lens = torch.IntTensor([len(l) for l in labels])

    max_len = max(len(l) for l in labels)
    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int64)
    for e, l in enumerate(labels):
        cat_labels[e, : len(l)] = l
    targets = torch.IntTensor(cat_labels)

    return targets, target_lens


# Tests -----------------------------------------------------------------------


def test_transducer_loss_small_worked_example_1() -> None:
    """Ensures TransducerLoss matches simple worked example.

    Consider three timesteps with perfectly trained model and exact alignments
    such that correct output should be:
    * a @ T=1
    * b @ T=2
    * blank @ T=3

    For alphabet = ['a', 'b', '<BLANK>'] this is a target sequence of:
    target = [torch.IntTensor([0, 1])] # == `ab`

    This would have probabilities:

    probs_t1 = torch.tensor([[1., 0, 0], [0, 0, 1], [0, 0, 1]])
    probs_t2 = torch.tensor([[0., 0, 1], [0, 1, 0], [0, 0, 1]])
    probs_t3 = torch.tensor([[0, 0, 1.], [0, 0, 1], [0, 0, 1]])

    In this case, expected loss == 0. BUT it is not possible to run this as
    log(0) = - `inf`.

    So consider value of `eps` such that at t = 1, u = 1 we have:
        probs_t1_u1 = torch.tensor([1. - 2 * eps, eps, eps])

    The loss values should monotonically increase as eps increases from
        ``0 -> 0.5``: this is what we will test.

    log_probs has shape: ``[batch = 1, max_seq_len = 3,
        max_output_seq_len + 1 = 3, vocab_size + 1 = 3)``

    Note that :py:class`TransducerLoss` typically takes *logits* as inputs
    rathee than log_probs (*and this is the recommended usage*) but the
    implementation of ``warprnnt_pytorch`` assumes the input is log_probs
    iff all values in the input are negative.
    """
    transducer_loss = TransducerLoss(blank=2, reduction="mean")
    targets = _label_collate_helper([torch.IntTensor([0, 1])])

    loss_values = []
    eps_values = [1e-6, 1e-4, 1e-2, 0.1, 0.2, 0.4, 0.49]
    for eps in eps_values:
        inputs = _get_log_probs_helper(eps)
        loss_value = transducer_loss(inputs, targets)

        assert isinstance(loss_value, torch.Tensor)
        loss_values.append(loss_value.item())

    loss_values_expected = loss_values.copy()
    loss_values_expected.sort()

    assert loss_values == loss_values_expected


def test_transducer_loss_small_worked_example_2() -> None:
    """Ensures TransducerLoss matches simple worked example.

    Consider untrained model (all probabilities equally likely) and the
    following conditions:
    * alphabet = ['a', <BLANK>']
    * (hence all probabilities are 0.5)
    * max_seq_len = 2 (i.e. x2 timesteps)
    * target sequence is y = [torch.IntTensor([0])] == 'a'

    There are two paths through the graph: (blank, a) or (a, blank). Each of
    these requires x3 transitions (as we must append blank at end) giving
    an expected total of ``0.5 ** 3 x 2 = 0.25``.

    Hence we have: ``expected_loss = - ln(0.25)``

    log_probs has shape: ``[batch = 1, max_seq_len = 2,
        max_output_seq_len + 1 = 2, vocab_size + 1 = 2)``

    Note that :py:class`TransducerLoss` typically takes *logits* as inputs
    rather than log_probs (*and this is the recommended usage*) but the
    implementation of ``warprnnt_pytorch`` assumes the input is log_probs
    iff all values in the input are negative.
    """
    transducer_loss = TransducerLoss(blank=1, reduction="mean")
    probs = torch.ones((1, 2, 2, 2)) * 0.5
    log_probs = probs.log()
    log_lens = torch.IntTensor([2])
    inputs = (log_probs, log_lens)

    targets = _label_collate_helper([torch.IntTensor([0])])

    loss_value = transducer_loss(inputs, targets).cpu()

    expected_loss = -torch.log(torch.tensor([1 / 4.0]))

    assert loss_value == expected_loss
