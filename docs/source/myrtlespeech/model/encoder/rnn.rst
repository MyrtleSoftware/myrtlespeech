=====
 rnn
=====

Python
------

The PyTorch LSTM, GRU, and RNN implementations are used.

.. autoclass:: torch.nn.LSTM
    :members:
    :show-inheritance:

.. autoclass:: torch.nn.GRU
    :members:
    :show-inheritance:

.. autoclass:: torch.nn.RNN
    :members:
    :show-inheritance:

Protobuf
--------

Message
~~~~~~~

.. literalinclude:: ../../../../../src/myrtlespeech/protos/rnn.proto
    :language: protobuf

Builder
~~~~~~~

.. automodule:: myrtlespeech.builders.rnn
    :members:
    :show-inheritance:
