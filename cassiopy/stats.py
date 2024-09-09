import numpy as np
import sklearn
import scipy.stats


class SkewT:
    """
    Generate synthetic data with skewness.
    """

    @staticmethod
    def rvs(mean, sigma, nu, lamb, n_samples=100):
        """
        Generate skew-t distribution.

        Parameters
        ==========
        mean : array-like
            The mean of the distribution with shape (n_dim).

        sigma : array-like
            The standard deviation of the distribution with shape (n_dim).

        nu : array-like
            The degree of freedom of the distribution with shape (n_dim).

        lamb : array-like
            The skewness of the distribution with shape (n_dim).

        n_samples : int
            Number of samples to generate.


        Returns
        =======
        data : array-like
            Generated data with shape (n_samples, n_dim).

        y_true : array-like
            Labels for the generated data with shape (n_samples,).

        Notes
        =====

        For more information, refer to the documentation :ref:`doc.stats.SkewT`

        Examples
        ========
        >>> from cassiopy.stats import Skew
        >>> sm = SkewT()
        >>> data = sm.rvs(mu=np.array([0, 0]), sigma=np.array([1, 1]), nu=np.array([10, 10]), lamb=np.array([0.5, 0.5]), n_samples=200)
        >>> data.shape
        (200, 2)

        """

        # Convert scalar inputs to arrays of the same shape if needed
        mean = np.asarray(mean)
        sigma = np.asarray(sigma)
        lamb = np.asarray(lamb)
        nu = np.asarray(nu)
        
        # Determine the dimensionality based on mean
        n_dim = mean.shape[0] if mean.ndim > 0 else 1    

        data = np.zeros((n_samples, n_dim))

        # data = mean + sigma * scipy.stats.skewnorm.rvs(
        #     a=lamb, loc=0, scale=1, size=(n_samples, n_dim)
        # ) / np.sqrt(
        #     scipy.stats.gamma.rvs(
        #         a=nu / 2, scale = 2 /nu, loc = (nu**2) / 4, size=(n_samples, n_dim)
        #     )
        # )

        data = mean + sigma * scipy.stats.skewnorm.rvs(
            a=lamb, loc=0, scale=1, size=(n_samples, n_dim)
        ) / np.sqrt(
            scipy.stats.gamma.rvs(
                a=nu / 2, scale = 2 /nu, loc = 0, size=(n_samples, n_dim)
            )
        )

        # data = mean + sigma * scipy.stats.skewnorm.rvs(
        #     a=lamb, loc=0, scale=1, size=(n_samples, n_dim)
        # ) / np.sqrt(
        #     scipy.stats.gamma.rvs(
        #         a=(nu/2), scale = (nu/2), loc = (nu**2) / 4, size=(n_samples, n_dim)
        #     )
        # )

        return data

    @staticmethod
    def random_cluster(
        n_samples=100, n_dim=1, n_clusters=4, random_state=None, labels=None
    ):
        """
        Generates random clusters based on the SkewT distribution.

        Parameters
        ==========
        n_samples : int
            Number of samples to generate.

        n_dim : int
            Number of dimensions.

        n_clusters : int
            Number of clusters.

        random_state : int
            Random seed for reproducibility.

        Returns
        =======
        data : array-like
            Generated data with shape (n_samples, n_dim).

        y_true : array-like
            Labels for the generated data with shape (n_samples,).

        Notes
        =====

        For more information, refer to the documentation :ref:`doc.stats.SkewT`

        Examples
        ========
        >>> from cassiopy.stats import Skew
        >>> sm = SkewT()
        >>> data, labels = sm.random_cluster(n_samples=200, n_dim=2, n_clusters=3, random_state=123)
        >>> data.shape
        (200, 2)
        >>> labels.shape
        (200,)

        """

        if random_state is None:
            random_state = np.random.randint(0, 2**32 - 1)

        np.random.seed(random_state)

        parametre = {
            "mu": np.random.uniform(-40, 40, size=(n_dim, n_clusters)),
            "sig": np.random.uniform(0.5, 5.0, size=(n_dim, n_clusters)),
            "nu": np.random.uniform(1.0, 10.0, size=(n_dim, n_clusters)),
            "lamb": np.random.uniform(-5, 5, size=(n_dim, n_clusters)),
            "alpha": np.full(n_clusters, 1.0 / n_clusters),
        }

        # Generate the number of samples for each cluster
        n = np.array([], dtype=int)

        cuts = sorted(np.random.choice(range(1, n_samples), size=n_clusters-1, replace=False))
        
        # Ajouter 0 au début et n_samples à la fin
        cuts = [0] + cuts + [n_samples]
        
        # Calculer la différence entre chaque coupure pour obtenir les n entiers
        for i in range(len(cuts)-1):
            n = np.append(n, cuts[i+1] - cuts[i])
        



        # Listes pour stocker les résultats
        data = np.zeros((np.sum(n), n_dim))

        # Boucle sur chaque dimension
        n_tot = 0

        for i in range(len(n)):
            data[n_tot : n_tot + n[i], :] = parametre["mu"][:, i] + parametre["sig"][
                :, i
            ] * scipy.stats.skewnorm.rvs(
                a=parametre["lamb"][:, i],
                loc=0,
                scale=1,
                size=(n[i], n_dim),
                random_state=random_state,
            ) / np.sqrt(
                scipy.stats.gamma.rvs(
                    a=parametre["nu"][:, i] / 2,
                    scale=2 / parametre["nu"][:, i],
                    loc = 0,
                    size=(n[i], n_dim),
                    random_state=random_state,
                )
            )
            n_tot += n[i]

        data = sklearn.utils.shuffle(data, random_state=random_state)

        if labels is not None:
            y_true = np.array([])
            for index, value in enumerate(n):
                y_true = np.append(y_true, (np.full(value, index)))

            y_true = sklearn.utils.shuffle(y_true, random_state=random_state)
            return data, y_true

        else:
            return data

    @staticmethod
    def pdf(x, mean, sigma, nu, lamb):
        """
        Probability density fonction

        Parameters
        ==========
        x : float
            The input data.

        mean : float
            The mean.

        sigma : float
            The standard deviation.

        nu : float
            The degree of freedom.

        lamb : float
            The skewness.

        Returns
        =======

        proba : float
            The probability density distribution of the data.

        References
        ==========
        [1] `Adelchi Azzalini, Antonella Capitanio, Distributions Generated by Perturbation of Symmetry with Emphasis on a Multivariate Skew t-Distribution, Journal of the Royal Statistical Society Series B: Statistical Methodology, Volume 65, Issue 2, May 2003, Pages 367–389. <https://doi.org/10.1111/1467-9868.00391>`_

        Notes
        =====

        For more information, refer to the documentation :ref:`doc.stats.SkewT`

        Examples
        ========

        >>> from cassiopy.stats import Skew
        >>> sm = SkewT()
        >>> x, mu, sigma, nu, lamb = 0.5, 0, 1, 10, 0.5
        >>> sm.pdf(x, mu, sigma, nu, lamb)
        0.3520653267642995
        """
        eta = (x - mean) / sigma
        A = lamb * eta * np.sqrt((nu + 1) / (nu + eta ** 2))
        B = scipy.stats.t.cdf(x = A, df = (nu + 1))
        C = scipy.stats.t.pdf(x = eta, df = nu)

        return 2 / sigma * C * B
