.. _doc.mixture.SkewMixture:


Mixture Model
=============


Skew-t Mixture models
----------------------

:ref:`SkewMixture` models a mixture of skew-t distributions. It provides various methods for working with skew-t distributed data, including generating samples and calculating densities.


**Examples:**

.. code-block:: python

    >>> import numpy as np
    >>> from cassiopy.mixture import SkewMixture
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> model = SkewMixture(n_cluster=2, n_iter=100, tol=1e-4, init='random')
    >>> model.fit(X)
    >>> model.mu
    array([[10.,  2.],
           [ 1.,  2.]])
    >>> model.predict([[0, 0], [12, 3]])
    array([1, 0])
    >>> model.predict_proba([[0, 0], [12, 3]])
    array([[0.99999999, 0.        ],
           [0.        , 0.90        ]])
    >>> model.save('model.h5')