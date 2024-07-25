.. _doc.stats.Skew:


Stats
=====

Skew-t distribution
-------------------

**Examples:**

.. code-block:: python

    >>> from cassiopy.stats import Skew
    >>> sm = Skew()
    >>> data, labels = sm.generate(n_samples=200, n_dim=2, n_clusters=3, random_state=123)
    >>> data.shape
    (200, 2)
    >>> labels.shape
    (200,)

Skew-t pdf
----------

**Examples:**

.. code-block:: python

    >>> from cassiopy.stats import Skew
    >>> sm = Skew()
    >>> x, mu, sigma, nu, lamb = 0.5, 0, 1, 10, 0.5
    >>> sm.pdf(x, mu, sigma, nu, lamb)
    0.3520653267642995   