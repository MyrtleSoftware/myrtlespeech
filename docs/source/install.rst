.. _install:

=========
 Install
=========

Development
------------

1. Setup environment:

   .. code-block:: bash

      conda env create -f environment.yml

2. Activate environment:

   .. code-block:: bash

      conda activate myrtlespeech

3. Install myrtlespeech:

  .. code-block:: bash

    pip install -e .


4. Install `pre-commit <https://pre-commit.com>`_ into git hooks.

   .. code-block:: bash

      pre-commit install

   The set of pre-commit git hooks listed in `.pre-commit-config.yaml` will run
   automatically before each commit. They can also be manually triggered:

   .. code-block:: bash

      pre-commit run --all-files

5. Compile the `Protocol Buffer
   <https://developers.google.com/protocol-buffers/>`_ files into Python
   modules:

   .. code-block:: bash

    protoc --proto_path src/ --python_out src/ src/myrtlespeech/protos/*.proto --mypy_out src/

   .. note::

        This command should be executed each time a ``.proto`` file is
        modified.


6. Install `NVIDIA Apex
   <https://github.com/NVIDIA/apex/tree/880ab925bce9f817a93988b021e12db5f67f7787>`_.
   As of 2019-08-21 it does _not_ have a Conda package and must be installed
   manually:

   .. code-block:: bash

    make deps/apex


7. Install `Warp-Transducer Loss <https://github.com/HawkAaron/warp-transducer/commit/c6d12f9e1562833c2b4e7ad84cb22aa4ba31d18c>`_:

  .. code-block:: bash

    export CUDA_HOME=/usr/local/cuda
    make deps/warp-transducer

  .. note::

       Replace ``/usr/local/cuda`` with your cuda location.

Continuous Integration
-----------------------

For CI, Conda is wrapped in a `Docker <https://www.docker.com>`_ container so
the package can be easily deployed to a cluster. To build the container:

.. code-block:: bash

   $ sudo docker build . -t myrtlespeech

The default Docker ``CMD`` is to run the full test suite under the ``ci``
Hypothesis profile (see :ref:`hypothesis-label`).

.. warning::

    The Dockerfile installs `NVIDIA Apex <https://github.com/NVIDIA/apex>`_,
    used for mixed precision, using a Python-only build and will omit some Apex
    features and performance improvements.
