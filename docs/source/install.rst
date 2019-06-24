.. _install:

=========
 Install
=========

Development
------------

1. Setup environment:

   .. code-block:: bash

      $ conda env create -f environment.yml

2. Install `pre-commit <https://pre-commit.com>`_ into git hooks.

   .. code-block:: bash

      $ pre-commit install

   The set of pre-commit git hooks listed in `.pre-commit-config.yaml` will run
   automatically before each commit. They can also be manually triggered:

   .. code-block:: bash

      $ pre-commit run --all-files

3. Compile the `Protocol Buffer
   <https://developers.google.com/protocol-buffers/>`_ files into Python
   modules:

   .. code-block:: bash

    $ protoc --proto_path src/ src/myrtlespeech/protos/*.proto --python_out=.

   .. note::

        This command should be executed each time a ``.proto`` file is
        modified.


Continuous Integration
-----------------------

For CI, Conda is wrapped in a `Docker <https://www.docker.com>`_ container so
the package can be easily deployed to a cluster. To build the container:

.. code-block:: bash

   $ sudo docker build . -t myrtlespeech

The default Docker ``CMD`` is to run the full test suite under the ``ci``
Hypothesis profile (see :ref:`hypothesis-label`).
