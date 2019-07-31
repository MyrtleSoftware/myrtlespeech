=========
 encoder
=========

The encoder part of an encoder-decoder model.

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    cnn_rnn_encoder
    vgg
    rnn


Python
------

.. automodule:: myrtlespeech.model.encoder_decoder.encoder.encoder
    :members:
    :show-inheritance:

Protobuf
--------

Message
~~~~~~~

.. literalinclude:: ../../../../../../src/myrtlespeech/protos/encoder.proto
    :language: protobuf

Builder
~~~~~~~

.. automodule:: myrtlespeech.builders.encoder
    :members:
    :show-inheritance:
