"""`VGG <https://arxiv.org/abs/1409.1556>`_ *"ConvNet"* implementations.

This code is based on PyTorch's VGG `implementation
<https://github.com/pytorch/vision/blob/ec203153095ad3d2e79fbf2865d80fe6076618fa/torchvision/models/vgg.py>`_
that is licensed under the BSD 3-Clause:

.. code-block:: none

    BSD 3-Clause License

    Copyright (c) Soumith Chintala 2016,
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from typing import List, Union

import torch.nn as nn


VGGConfig = List[Union[int, str]]


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "M",
    ],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}
r"""VGG *"ConvNet"* configurations as defined in the `paper`_.

.. _paper: https://arxiv.org/abs/1409.1556

    >>> for name, cfg in cfgs.items():
    ...     print(name, cfg)
    A [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    B [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    D [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    E [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
"""  # pylint: disable=W0105


def make_layers(
    cfg: VGGConfig,
    in_channels: int = 3,
    batch_norm: bool = False,
    initialize_weights: bool = True,
):
    """Returns a :py:class:`torch.nn.Sequential` module for the given `cfg`.

    Example:
        >>> cfgs['A']
        [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        >>> make_layers(cfgs['A'], in_channels=5, batch_norm=True)
        Sequential(
          (0): Conv2d(5, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace)
          (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): ReLU(inplace)
          (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (8): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (9): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (10): ReLU(inplace)
          (11): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (12): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (13): ReLU(inplace)
          (14): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (15): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (16): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (17): ReLU(inplace)
          (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (19): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (20): ReLU(inplace)
          (21): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (22): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (23): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (24): ReLU(inplace)
          (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (26): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (27): ReLU(inplace)
          (28): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )


    Args:
        cfg: A *"ConvNet"* configuration specified as a list of :py:class:`int`
            and :py:class:`str`.

            Each :py:class:`int` will be converted to a
            :py:class:`torch.nn.Conv2d` where the :py:class:`int` denotes the
            number of ``out_channels``. The other :py:class:`torch.nn.Conv2d`
            parameters are set the defaults except ``kernel_size=3, padding=1``
            and ``in_channels`` equals the previous :py:class:`int` in the list
            or the ``in_channels`` argument if the first :py:class:`int`.

            :py:class:`torch.nn.ReLU` layers with ``inplace=True`` will be
            inserted after each :py:class:`torch.nn.Conv2d`.

            Each :py:class:`str` must be equal to `'M'` and will be converted
            to a :py:class:`torch.nn.MaxPool2d` layer with ``kernel_size=2,
            stride=2``.

        in_channels: ``in_channels`` argument for the first
            :py:class:`torch.nn.Conv2d``.

        batch_norm: If :py:data:`True` then a :py:class:`torch.nn.BatchNorm2d`
            layer is inserted after each :py:class:`torch.nn.Conv2d`.

        initialize_weights: If :py:data:`True`, :py:class:`torch.nn.Conv2d`
            weights are initialized using
            :py:func:`torch.nn.init.kaiming_normal_` with ``mode="fan_out",
            nonlinearity="relu"`` and biases initialized to 0.
            :py:class:`torch.nn.BatchNorm2d` weights (scales) are set to 1 and
            biases 0.

            If :py:data:`False` the default initialization for each PyTorch
            class is used.


    Returns:
        A :py:class:`torch.nn.Sequential` :py:class:`torch.nn.Module`
        containing the :py:class:`torch.nn.Conv2d`,
        :py:class:`torch.nn.BatchNorm2d` and :py:class:`torch.nn.ReLU`
        :py:class:`torch.nn.Module`'s.

    Raises:
        :py:class:`ValueError`: If ``cfg`` contains values other than ``"M"``
            or :py:class:`int`.
    """
    layers: List[nn.Module] = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif isinstance(v, int):
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        else:
            raise ValueError("unknown value %r in cfg" % v)

    if initialize_weights:
        for m in layers:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    return nn.Sequential(*layers)
