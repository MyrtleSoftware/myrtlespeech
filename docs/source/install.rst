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
