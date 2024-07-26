.. _doc.mixture:


Mixture Model
=============


Skew-t Mixture models
----------------------

Skew-t mixture models are an unsupervised machine learning method used for clustering data. These models extend Gaussian Mixture Models (GMM) to accommodate non-symmetric distributions by employing skew-t distributions.

.. math::
       Y = \mu + \sigma \frac{Z}{\sqrt{\tau}}, \qquad Z\sim\mathcal{SN}(\lambda), \qquad \tau\sim\Gamma\left(\frac{\nu}{2}, \frac{\nu}{2}\right) 

With 

:math:`\mu` : location parameter

:math:`\sigma` : scale parameter

:math:`\nu` : degrees of freedom

:math:`\lambda` : skewness parameter

:math:`\Gamma` : gamma distribution

A skew-t mixture model assumes that the data is generated from a finite mixture of skew-t distributions, each characterized by unknown parameters. This approach is particularly useful for modeling data with skewed distributions, providing a more flexible and accurate representation than traditional GMMs.

.. math::
   p(\vec{x_i};\vec{\theta_{k}})  = \sum_{k=1}^{K} \alpha_k  \; p(\vec{x_i}|\vec{\theta_{k}}) 

With 

.. math::
   p(x_{ij} \mid \vec{y_i};\theta) = \left(\prod_{k=1}^K \mathcal{ST}(x_{ij} \mid \mu_{kj}, \sigma_{kj}^2, \lambda_{kj}, \nu_{kj})^{y_{ik}}\right)

Where

:math:`K` : number of clusters

:math:`\alpha_k` : mixing coefficient

:math:`y_{ik}` : vector of size :math:`K` (number of clusters) whose value takes 1 if the data :math:`i` belongs to cluster :math:`k`, 0 otherwise

:math:`\theta_k` : parameters of the :math:`k`-th cluster

:math:`\mathcal{ST}` : skew-t probabibility density function


**Examples:**

.. code-block:: python

    >>> import numpy as np
    >>> from cassiopy.mixture import SkewTMixture
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> model = SkewTMixture(n_cluster=2, n_iter=100, tol=1e-4, init='random')
    >>> model.fit(X)
    >>> model.mu
    array([[10.,  2.],
           [ 1.,  2.]])
    >>> model.predict([[0, 0], [12, 3]])
    array([0, 1])
    >>> model.predict_proba([[0, 0], [12, 3]])
    array([[0.99999999, 0.        ],
           [0.10      , 0.90      ]])
    >>> model.save('model.h5')

**See also**

:func:`Skew-t Mixture <cassiopy.mixture.SkewTMixture>`


Skew-t Uniform Mixture models
------------------------------

Skew-t uniform mixture models are an unsupervised machine learning method used for clustering data with an uniform background. These models extend Gaussian Mixture Models (GMM) to accommodate non-symmetric distributions by employing skew-t distributions with a uniform background.

.. math::
   p(\vec{x_i};\vec{\theta_{k}})  = \sum_{k=1}^{K} \alpha_k  \; p(\vec{x_i}|\vec{\theta_{k}}) + \alpha_{k+1} \frac{1}{V}


**Examples:**

.. code-block:: python

    >>> import numpy as np
    >>> from cassiopy.mixture import SkewTUniformMixture
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> model = SkewTUniformMixture(n_cluster=2, n_iter=100, tol=1e-4, init='random')
    >>> model.fit(X)
    >>> model.mu
    array([[10.,  2.],
           [ 1.,  2.]])
    >>> model.predict([[0, 0], [12, 3]])
    array([0, 1])
    >>> model.predict_proba([[0, 0], [12, 3]])
    array([[0.99999999, 0.        , 0.        ],
           [0.        , 0.90      , 0.10      ]])
    >>> model.save('model.h5')

**See also**

:func:`Skew-t Mixture <cassiopy.mixture.SkewTUniformMixture>`
