======
 Test
======

The `pytest <https://docs.pytest.org>`_ framework handles testing, the
`hypothesis <https://hypothesis.readthedocs.io>`_ library is used for
property-based tests, and the `doctest
<https://docs.python.org/3.7/library/doctest.html>`_ module is used for testing
interactive Python sessions within the documentation (docstrings). The
`pytest-cov <https://pytest-cov.readthedocs.io/en/latest/index.html>`_
``pytest`` plugin provides coverage reports.

The ``pytest.ini`` configuration file specifies the default test options.

To run the tests:

.. code-block:: bash

   $ pytest

.. _hypothesis-label:

Hypothesis
-----------

Three ``hypothesis`` profiles are defined with different settings.

The first, ``single``, runs only a single test example for each test. This
should **not** be used to validate functionality! It is useful for, amongst
other things, checking coverage is a quick way.

The second, ``dev``, is the default profile and is meant for development. It
runs a reasonable number of examples whilst still facilitating quick iteration
times.

The third, ``ci``, increases the max number of examples and max time per
example compared to ``dev`` and is meant for the continuous integration system
where correctness, rather than iteration time, is more crucial.

See :file:`tests/__init__.py`.

The profile can be changed by setting the ``HYPOTHESIS_PROFILE`` environment
variable to the desired profile. For example, to run the tests locally with the
``ci`` environment:

.. code-block:: bash

    $ HYPOTHESIS_PROFILE=ci pytest
