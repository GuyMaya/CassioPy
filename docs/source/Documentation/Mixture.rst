.. _doc.mixture:

.. meta::
   :description: Documentation of Skew-t Mixture Models in CassioPy
   :keywords: skew-t, clustering, mixture models, machine learning

Mixture Model
=============


.. _doc.mixture.SkewTMixture:

Skew-t Mixture Models
----------------------

Skew-t mixture models are an unsupervised machine learning method used for clustering data. These models extend Gaussian Mixture Models (GMM) to accommodate non-symmetric distributions by employing skew-t distributions.

A random variable X follows a skew-t distribution if it can be represented by:

.. math::
       X = \mu + \sigma \frac{U}{\sqrt{\tau}}, \qquad with  \qquad U\sim\mathcal{SN}(\lambda), \qquad \tau\sim\Gamma\left(\frac{\nu}{2}, \frac{\nu}{2}\right) 

with 

:math:`\mu \in \mathbb{R}` : location parameter

:math:`\sigma \in \mathbb{R^*_+}` : scale parameter, diagonal covariance matrix

:math:`\nu \in \mathbb{R^*_+}` : degrees of freedom

:math:`\lambda \in \mathbb{R}` : skewness parameter

:math:`\Gamma(\alpha, \beta)` : gamma distribution with shape parameter :math:`\alpha` and an inverse scale parameter :math:`\beta`

:math:`\mathcal{SN}` : standard normal distribution with parameter :math:`\lambda`

:math:`\mathcal{SN}(x) = 2\phi(x)\Phi(\lambda x)` with :math:`\phi` the standard normal density and :math:`\Phi` the standard normal cumulative distribution function


A skew-t mixture model assumes that the data is generated from a finite mixture of skew-t distributions, each characterized by unknown parameters. This approach is particularly useful for modeling data with skewed distributions, providing a more flexible and accurate representation than traditional GMMs. 

For sake of simplicity, we assume that variables are independent given the cluster. 

.. math::
   p(\vec{x_i};\vec{\theta_{k}})  = \sum_{k=1}^{K} \alpha_k \prod_{j=1}^d \; p(x_{ij} \mid \vec{\theta}_k)
   
Where :math:`\vec{\theta}` groups all the parameters, :math:`\alpha_k` is the proportion of the :math:`k`-th cluster, and :math:`\vec{\theta}_k` are parameters related to the cluster :math:`k`.


.. math::
   p(x_{ij} \mid \vec{\theta}_k) = \mathcal{ST}(x_{ij} \mid \mu_{kj}, \sigma_{kj}, \lambda_{kj}, \nu_{kj})

Where

:math:`K` : number of clusters

:math:`d` : number of features

:math:`\mathcal{ST}` : skew-t probabibility density function


**Examples:**

.. code-block:: python

    >>> import numpy as np
    >>> from cassiopy.mixture import SkewTMixture
    >>> from cassiopy.mixture import SkewT
    >>> data, y_true = SkewT().random_cluster(n_samples=10000, n_dim=2, n_clusters=10, labels=True, random_state=4)

    >>> model = SkewTMixture(n_cluster=10, init='gmm', n_iter=100, n_init=4, verbose=0).fit(data)
    >>> y_pred = model.predict(data)
    >>> model.ARI(y_true, y_pred)
   0.9828358328555337
    >>> model.save('model.h5')

    >>> # Plotting
    >>> plt.scatter(data[:, 0], data[:, 1], c=y_pred, cmap='viridis', s=3)
    >>> plt.xlabel('dim 1')
    >>> plt.ylabel('dim 2')

    >>> plt.text(max(data[:, 0]), max(data[:, 1]), s = f'ARI: {MM.ARI(y_true, y_pred):.3f}', fontsize=12, color='black', ha='right', va='top')
    >>> plt.title('Clustering using SkewT Mixture Model')



.. figure:: ../_static/Images/Skewt_clustering.png
   :alt: Clustering using SkewT Mixture Model
   :width: 500px
   :align: center


.. dropdown:: References


   .. bibliography:: referencemixturemodel.bib
      :all:


**See also**

:func:`Skew-t Mixture <cassiopy.mixture.SkewTMixture>`



.. _doc.mixture.SkewTUniformMixture:

Skew-t Uniform Mixture Models
------------------------------

The Skew-t uniform mixture models is an unsupervised machine learning method used for clustering data into groups following skewed distributions (see :numref:`doc.mixture.SkewTMixture` above) together with an uniform background. These models extend Gaussian Mixture Models (GMM) to accommodate non-symmetric distributions by employing skew-t distributions with a uniform background.

.. math::
   p(\vec{x_i};\vec{\theta})  = \sum_{k=1}^{K} \alpha_k  \; p(\vec{x_i}|\vec{\theta_{k}}) + \alpha_{K+1} \frac{1}{V}

Where :math:`V` is the volume of the uniform background.

.. math::
       V = \prod_{j=1}^d \left( x_{\max,j} - x_{\min,j} \right)

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


.. _doc.mixture.BIC:

Bayesian Information Criterion (BIC)
------------------------------------

The Bayesian Information Criterion (BIC) is a criterion for model selection among a finite set of models. 
The model with the lowest BIC is preferred. The BIC is defined as:

.. math::
   BIC = -2 \log(L) + p \log(n)

Where

:math:`L` : likelihood of the model

:math:`p` : number of parameters in the model

:math:`n` : number of samples


.. dropdown:: References

   .. bibliography:: referenceBIC.bib
      :all:




.. _doc.mixture.ARI:

Adjusted Rand Index (ARI)
--------------------------

The Adjusted Rand Index (ARI) is a measure of the similarity between two data clusterings. It ensure that random clusterings receive a score close to zero, with a maximum score of 1 indicating perfect agreement between the clusterings.
The Rand Index is defined as:

.. math::
   RI = \frac{a + b}{\binom{N}{2}}

With 

:math:`a` : number of pairs of elements that are in the same cluster in both the true and predicted clusters

:math:`b` : number of pairs of elements that are in different clusters in both the true and predicted clusters

:math:`\binom{N}{2}` : number of possible pairs of elements

Value attended for a random clustering :

.. math::
   E = \frac{\sum_i \binom{n_i}{2} \quad \sum_j \binom{n_j}{2}}{\binom{N}{2}}

:math:`n_i` : number of elements in the :math:`i`-th cluster in the true clustering

:math:`n_j` : number of elements in the :math:`j`-th cluster in the predicted clustering

Adjusted Rand Index :

.. math::
   ARI = \frac{RI - E}{max(RI) - E}

With :math:`max(RI) = \frac{1}{2} \left(\sum_i\binom{n_i}{2} + \sum_j\binom{n_j}{2} \right)` the maximum possible value of the Rand Index

 **Special Cases:**
   - When :math:`ARI=1` , the two clusterings are identical, perfect agreement
   - When :math:`ARI=0` , the two clusterings are random, no agreement
   - When :math:`ARI=-1` , the two clusterings are different, perfect disagreement

.. dropdown:: References


   .. bibliography:: referenceARI.bib
      :all:


