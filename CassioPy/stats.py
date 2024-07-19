import numpy as np
import sklearn
import scipy.stats

class Skew:
    """
    Generate synthetic data with skewness.

    Parameters:

    - n_samples :int
        Number of samples to generate.

    - n_dim : int
        Number of dimensions.
        
    - n_clusters : int
        Number of clusters.

    - random_state : int
        Random seed for reproducibility.

    Returns:

    - data : ndarray
        Generated data with shape (n_samples, n_dim).

    - y_true : ndarray
        Labels for the generated data with shape (n_samples,).

    Example:

    >>> from cassiopy.stats import Skew
    >>> skew_set = Skew()
    >>> data, y_true = skew_set.generate(n_samples=100, n_dim=1, n_cluster=4, random_state=42)


    """

    def __init__(self):
        return

    def generate(self, n_samples=100, n_dim=1, n_clusters=4, random_state=42, labels=None):
        """
        Generate synthetic data with specified parameters.

        Parameters:

        - n_samples : int
            Number of samples to generate.

        - n_dim : int
            Number of dimensions.

        - n_clusters : int
            Number of clusters.

        - random_state : int
            Random seed for reproducibility.

        Returns:

        - data : ndarray
            Generated data with shape (n_samples, n_dim).

        - y_true : ndarray
            Labels for the generated data with shape (n_samples,).
        """
        parametre = {
            'mu' : np.random.uniform(-50, 50, size=(n_dim, n_clusters)),
            'sig' : np.random.uniform(0.5, 10., size=(n_dim, n_clusters)), 
            'nu' : np.random.uniform(1., 10., size=(n_dim, n_clusters)), 
            'lamb' : np.random.uniform(-5., 5., size=(n_dim, n_clusters)),
            'alpha' : np.full(n_clusters, 1.0 / n_clusters)
        }

        n = np.array([], dtype=int)
        for i in range(n_clusters-1):
            if i == 0:
                n = np.append(n, np.random.randint(1, n_samples))
            else:
                n = np.append(n, np.random.randint(1, n_samples - np.sum(n)))

        n = np.append(n, n_samples - np.sum(n)) 

        # Listes pour stocker les résultats
        data = np.zeros((np.sum(n), n_dim))

        # Boucle sur chaque dimension
        n_tot = 0

        for i in range(len(n)):
            data[n_tot:n_tot+n[i], :] = (parametre['mu'][:, i] + 
                    parametre['sig'][:, i] * scipy.stats.skewnorm.rvs(a=parametre['lamb'][:, i], loc=0, scale=1, size=(n[i], n_dim), random_state=random_state) /
                    np.sqrt(scipy.stats.gamma.rvs(a=parametre['nu'][:, i]/2, scale=2/parametre['nu'][:, i], size=(n[i], n_dim), random_state=random_state)))
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

    def pdf(self, x, mu, sigma, nu, lambda_):
        """densite de la skew t
        Implementation de l'equation (3) de Lin2007

        Parameters:

        - x : float
            The input data.
        - mu : float
            The mean of the cluster.
        - sigma : float
            The standard deviation of the cluster.
        - nu : float
            The degree of freedom of the cluster.
        - lambda_ : float
            The skewness of the cluster.

        Returns:

        - proba : float
            The probability of the data.
            
        """
        eta = (x - mu) / sigma
        A = lambda_ * eta * np.sqrt((nu + 1) / (nu + eta ** 2))
        B = scipy.stats.t.cdf(A, nu + 1)
        C = scipy.stats.t.pdf(eta, nu)

        return 2 / sigma * C * B