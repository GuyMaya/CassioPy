import numpy as np
import os
import scipy
from cassiopy.stats import Skew

class SkewTMixture:
    """
    Skew-t Mixture Model for clustering using PX-EM algorithm.

    Parameters
    ==========
    n_cluster : int
        The number of mixture components (clusters).

    n_iter : int, default=10
        The number of iterations to perform during the parameter estimation.

    tol : float, default=1e-5
        The convergence threshold. Iterations will stop when the
        improvement is below this threshold.

    init : {'random', 'kmeans', 'gmm', 'params'}, default='random'
        The method used to initialize the parameters.
        Must be one of:

        - 'random': Parameters are initialized randomly.
        - 'params': User-provided parameters are used for initialization.
        - 'kmeans': Parameters are initialized using K-means.
        - 'gmm': Parameters are initialized using a Gaussian Mixture Model.

    params : dict, default=None
        The user-provided initial parameters. Used only if `init` is 'params'.

    References
    ==========

    [1] `Lin, Tsung & Lee, Jack & Hsieh, Wan. (2007). Robust mixture models using the skew-t distribution. Statistics and Computing. 17. 81-92. 10.1007/s11222-006-9005-8. <https://doi.org/10.1007/s11222-006-9005-8>`_

    Notes
    =====

    For more information, refer to the documentation:ref:`doc.mixture.SkewTMixture`

    Examples
    ========
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
    """

    def __init__(
        self, n_cluster: int, n_iter=10, tol=1e-8, init="gmm", params=None, n_init_gmm=8
    ):
        self.n_cluster = n_cluster
        self.n_iter = n_iter
        self.tol = tol
        self.init_method = init
        if self.init_method == "params":
            self.params = params

        if self.init_method == "gmm":
            self.n_init_gmm = n_init_gmm

    def fit(self, X):
        """
        Fits the SkewMM model to the input data.

        Parameters
        ==========
        X : array-like 
            Shape (n_samples, n_features). Each sample is represented by a feature vector Xi = [e_i, i_i, H_i, a_i].
        """

        # Implementation of the fit method
        if self.init_method == "random":
            best_params = self.initialisation_random(X)
            self.initialisation_params(best_params, X)

        elif self.init_method == "kmeans":
            best_params = self.initialisation_kmeans(X)
            self.initialisation_params(best_params, X)

        elif self.init_method == "params":
            self.initialisation_params(self.params, X)

        elif self.init_method == "gmm":
            best_params = None
            best_LL = -np.inf

            params_to_test = [
                self.initialisation_gmm(X) for _ in range(self.n_init_gmm)
            ]

            # Test each set of parameters and keep the best one
            for params in params_to_test:
                self.initialisation_params(params, X)
                self.p = self.phi(X)
                LL = self.LL(X)
                if LL > best_LL:
                    best_LL = LL
                    best_params = params

            # Apply the best parameters found
            self.initialisation_params(best_params, X)

        elif self.init_method == "likelihood":
            param_random = self.initialisation_random(X)
            LLN_random = self.E_step(X)
            param_gmm = self.initialisation_gmm(X)
            LLN_gmm = self.E_step(X)

            if LLN_random > LLN_gmm:
                self.initialisation_params(param_random, X)
                print("initialization random")
            else:
                self.initialisation_params(param_gmm, X)
                print("initialization gmm")
        else:
            raise ValueError(
                f"Error: The initialization method {self.init_method} is not recognized, please choose from 'random', 'kmeans', 'params', 'gmm' ou 'likelihood'"
            )

        print("initialization method :", self.init_method)

        self.E_log_likelihood = [-np.inf]

        self.save("track_0")

        i = 0

        while i < self.n_iter + 1:
            print(f"iteration: {i}/{self.n_iter}")

            self.E_step(X)
            E_log_likelihood_new = self.LL(X)

            if np.abs(E_log_likelihood_new - self.E_log_likelihood[i]) <= self.tol:
                if i < 2:
                    print("reinitialization not enough iteration")
                    return self.fit(X)
                else:
                    break

            if E_log_likelihood_new < self.E_log_likelihood[i]:
                if i < 2:
                    print("reinitialization not enough iteration")
                    return self.fit(X)
                else:
                    break

            self.E_log_likelihood.append(E_log_likelihood_new)
            self.M_step(X)

            if np.any(np.isnan(self.sig)):
                if i < 2:
                    print("reinitialization : sig nan")
                    return self.fit(X)
                else:
                    self.load(f"Models_folder/track_{i-1}.h5")
                    self.n_iter = i - 1
                    break

            if np.any(self.nu < 0):
                if i < 2:
                    print("reinitialization : nu <0")
                    return self.fit(X)
                else:
                    self.load(f"Models_folder/track_{i-1}.h5")
                    self.n_iter = i - 1
                    break

            if np.any(np.diagonal(self.sig, axis1=0, axis2=1) < 0):
                if i < 2:
                    np.diagonal(self.sig, axis1=0, axis2=1)

                    print("sig", self.sig)

                    print("eig:", np.diagonal(self.sig, axis1=0, axis2=1))
                    print("reinitialization : negative equity")
                    return self.fit(X)
                else:
                    model = self.load(f"Models_folder/track_{i-1}.h5")
                    self.n_iter = i - 1
                    break

            self.save(f"track_{i}")

            i += 1

        return self

    def initialisation_random(self, X):
        """
        Random initialization method for the SkewMM algorithm.

        Parameters
        ==========
        X : array-like
            Input data array.

        Returns
        =======
        dict : dict
            A dictionary containing the initialized parameters:

            - 'mu': numpy array
                Matrix of means.

            - 'sig': numpy array
                Matrix of covariances.

            - 'nu': numpy array
                Matrix of degrees of freedom.

            - 'lamb': numpy array
                Matrix of skewness parameters.

            - 'alpha': numpy array
                Array of cluster proportions.
        """
        # Implementation of the random initialization method
        max_x = np.max(X, axis=0)[:, np.newaxis] if X.ndim > 1 else np.max(X)

        # initialization of the average matrix
        mu = np.random.rand(X.shape[1], self.n_cluster) * max_x

        # initialization of the covariance matrix
        sig = np.ones((X.shape[1], self.n_cluster))

        # initialization of the degree of freedom
        nu = np.random.rand(X.shape[1], self.n_cluster)

        # initialization of the skewness parameter
        lamb = np.random.uniform(low=1e-6, high=5.0, size=(X.shape[1], self.n_cluster))

        # initialization of the prior, proportion of data in cluster k
        alpha = np.random.rand(self.n_cluster)

        return {"mu": mu, "sig": sig, "nu": nu, "lamb": lamb, "alpha": alpha}

    def initialisation_params(self, params, X):
        """
        Initialize the parameters of the SkewMM model.

        Parameters
        ==========
        params : dict
            A dictionary containing the initial values for the model parameters.

            - 'mu' : ndarray
                The mean vectors for each cluster. Shape: (n_features, n_cluster).

            - 'sig' : ndarray
                The covariance matrices for each cluster. Shape: (n_features, n_cluster).

            - 'nu' : ndarray
                The degrees of freedom for each cluster. Shape: (n_features, n_cluster).

            - 'lamb' : ndarray
                The skewness parameters for each cluster. Shape: (n_features, n_cluster).

            - 'alpha' : ndarray
                The mixing proportions for each cluster. Shape: (n_cluster,).

        X : ndarray
            The input data. Shape: (n_samples, n_features).
        """
        if params["mu"].shape != (X.shape[1], self.n_cluster):
            raise ValueError(
                f"Error: The size of the matrix must be {(X.shape[1], self.n_cluster)}, but it is {params['mu'].shape}"
            )
        self.mu = np.array(params["mu"], dtype=float)

        if params["sig"].shape != (X.shape[1], self.n_cluster):
            raise ValueError(
                f"Error: The size of the matrix must be {(X.shape[1], self.n_cluster)}, but it is {params['sig'].shape}"
            )
        self.sig = np.array(params["sig"], dtype=float)

        if params["nu"].shape != (X.shape[1], self.n_cluster):
            raise ValueError(
                f"Error: The size of the matrix must be {(X.shape[1], self.n_cluster)}, but it is {params['nu'].shape}"
            )
        self.nu = np.array(params["nu"], dtype=float)

        if params["lamb"].shape != (X.shape[1], self.n_cluster):
            raise ValueError(
                f"Error: The size of the matrix must be {(X.shape[1], self.n_cluster)}, but it is {params['lamb'].shape}"
            )
        self.lamb = np.array(params["lamb"], dtype=float)

        if params["alpha"].shape != (self.n_cluster,):
            raise ValueError(
                f"Error: The size of the matrix must be {(self.n_cluster)}, but it is {params['alpha'].shape}"
            )
        self.alpha = np.array(params["alpha"], dtype=float)

        return self

    def initialisation_kmeans(self, X, default_n_init="auto"):
        """
        Initializes the parameters for the SkewMM algorithm using the K-means initialization method.

        Parameters
        ==========
        X : ndarray
            The input data matrix of shape (n_samples, n_features).

        default_n_init : int, default='auto'
            The number of times the K-means algorithm will be run with different centroid seeds. Default is 'auto'.

        Returns
        =======
        dict : dict
            A dictionary containing the initialized parameters:

            - 'mu': numpy array
                Matrix of means.

            - 'sig': numpy array
                Matrix of covariances.

            - 'nu': numpy array
                Matrix of degrees of freedom.

            - 'lamb': numpy array
                Matrix of skewness parameters.

            - 'alpha': numpy array
                Array of cluster proportions.
        """
        from sklearn.cluster import KMeans

        # Implementation of the K-means initialization method
        # Use KMeans to get the cluster centers
        kmeans = KMeans(
            n_clusters=self.n_cluster, n_init=20
        )  # pas de cluster pour le bruit
        kmeans.fit(X)
        cluster_centers = kmeans.cluster_centers_

        # Initializations of the mean matrix
        mu = cluster_centers.T

        # Initializations of the covariance matrix
        sig = np.ones((X.shape[1], self.n_cluster))

        # Initializations of the degree of freedom
        nu = np.random.rand(X.shape[1], self.n_cluster)

        # Initializations of the skewness parameter
        lamb = np.random.uniform(low=1e-6, high=10.0, size=(X.shape[1], self.n_cluster))

        # Initializations of the prior, proportion of data in cluster k
        alpha = np.random.rand(self.n_cluster)
        self.alpha = self.alpha / np.sum(self.alpha)

        return {"mu": mu, "sig": sig, "nu": nu, "lamb": lamb, "alpha": alpha}

    def initialisation_gmm(self, X):
        """
        Initialize the parameters for the Gaussian Mixture Model (GMM).

        Parameters
        ==========
        X : array-like
            Input data matrix of shape (n_samples, n_features).

        Returns
        =======
        dict : dict
            A dictionary containing the initialized parameters:

            - 'mu': numpy array
                Matrix of means.

            - 'sig': numpy array
                Matrix of covariances.

            - 'nu': numpy array
                Matrix of degrees of freedom.

            - 'lamb': numpy array
                Matrix of skewness parameters.

            - 'alpha': numpy array
                Array of cluster proportions.
        """
        from sklearn.mixture import GaussianMixture

        # Use the Gaussian Mixture Model to initialize the parameters
        gmm = GaussianMixture(
            n_components=self.n_cluster,
            covariance_type="diag",
            reg_covar=1e-6,
            tol=1e-6,
        )
        gmm.fit(X)

        # Initializations of the mean matrix with the means of the GMM
        mu = gmm.means_.T

        # Initializations of the covariance matrix with the covariances of the GMM
        sig = gmm.covariances_.T

        # Initialisations of the degree of freedom
        nu = np.random.uniform(0, 10, size=(X.shape[1], self.n_cluster))

        # Initialisations of the skewness parameter
        lamb = np.random.uniform(low=1e-6, high=1e-5, size=(X.shape[1], self.n_cluster))

        # Initializations of the prior, proportion of data in cluster k
        alpha = gmm.weights_

        return {"mu": mu, "sig": sig, "nu": nu, "lamb": lamb, "alpha": alpha}

    def g_of_nu(self, x, nu):
        """
        Calculate the value of g(nu) for a given x and nu.

        Parameters
        ==========
        x : float
            The input value.
        
        nu : float
            The parameter value.
        """
        A = scipy.special.digamma((nu + 2) / 2)
        B = -scipy.special.digamma((nu + 1) / 2)
        C = -np.log(1 + (x**2) / (nu + 1))
        D = ((nu + 1) * x**2 - nu - 1) / ((nu + 1) * (nu + 1 + x**2))
        return A + B + C + D

    def phi(self, x):
        """
        Calculates the probability density function (PDF) for each data point in x.

        Parameters
        ==========
        x : array-like
            Shape : (n_samples, n_features). The input data points.

        Returns
        =======
        p : array-like 
            Shape : (n_samples, n_cluster). The PDF values for each data point in x.
        """
        p = np.zeros((x.shape[0], self.n_cluster))
        for index, value in enumerate(x):
            p[index, :] = self.alpha[:]
            for dim in range(x.shape[1]):
                p[index, :] *= Skew.pdf(
                    x = value[dim],
                    mu = self.mu[dim, :],
                    sigma = self.sig[dim, :],
                    nu = self.nu[dim, :],
                    lamb = self.lamb[dim, :]
                )

        return p

    def E_step(self, X):
        """
        Performs the E-step of the SkewMM algorithm.
        """

        # Implementation of the predict_proba method
        self.p = self.phi(X)

        self.tik = self.p / np.sum(self.p, axis=1)[:, np.newaxis]

        # Calculation of s1
        nu_expended = self.nu[np.newaxis, :, :]

        self.eta = (X[:, :, np.newaxis] - self.mu[np.newaxis, :, :]) / self.sig[
            np.newaxis, :, :
        ]

        M = (
            self.lamb[np.newaxis, :, :]
            * self.eta
            * np.sqrt((self.nu[np.newaxis, :, :] + 1) / (nu_expended + (self.eta) ** 2))
        )
        T3_cdf = scipy.stats.t.cdf(
            M * np.sqrt((nu_expended + 3) / (nu_expended + 1)), df=(nu_expended + 3)
        )
        T1_cdf = scipy.stats.t.cdf(M, df=(nu_expended + 1))

        self.s1 = (
            self.tik[:, np.newaxis, :]
            * ((nu_expended + 1) / (nu_expended + (self.eta) ** 2))
            * (T3_cdf / T1_cdf)
        )

        # Calculation of s2
        self.delta = self.lamb / np.sqrt(1 + self.lamb**2)
        term1 = (
            self.delta[np.newaxis, :, :]
            * (X[:, :, np.newaxis] - self.mu[np.newaxis, :, :])
            * self.s1
        )

        f = np.zeros((X.shape[0], X.shape[1], self.n_cluster))

        for dim in range(X.shape[1]):
            f[:, dim, :] = Skew.pdf(
                X[:, dim][:, np.newaxis],
                self.mu[dim, :],
                self.sig[dim, :],
                self.nu[dim, :],
                self.lamb[dim, :],
            )

        term2 = np.sqrt(1 - self.delta[np.newaxis, :, :] ** 2) / (np.pi * f[:, :, :])

        term3 = (self.eta**2) / (
            nu_expended * (1 - (self.delta[np.newaxis, :, :]) ** 2)
        )
        term4 = (term3 + 1) ** (-((nu_expended / 2) + 1))
        self.s2 = term1 + self.tik[:, np.newaxis, :] * term2 * term4

        # Calculation of s3
        partie1 = (
            self.delta[np.newaxis, :, :] ** 2
            * (X[:, :, np.newaxis] - self.mu[np.newaxis, :, :]) ** 2
            * self.s1
        )

        partie2 = (1 - self.delta[np.newaxis, :, :] ** 2) * (
            self.sig[np.newaxis, :, :] ** 2
        )

        partie3 = (
            self.delta[np.newaxis, :, :]
            * (X[:, :, np.newaxis] - self.mu[np.newaxis, :, :])
            * np.sqrt(1 - self.delta[np.newaxis, :, :] ** 2)
        ) / (np.pi * f)

        partie4 = self.eta**2 / (
            nu_expended * (1 - (self.delta[np.newaxis, :, :]) ** 2)
        )
        partie5 = (partie4 + 1) ** (-((self.nu[np.newaxis, :, :] / 2) + 1))

        self.s3 = partie1 + self.tik[:, np.newaxis, :] * (partie2 + partie3 * partie5)

        # Calculation of s4
        P = np.log((self.eta**2 + nu_expended) / 2)

        Q = (nu_expended + 1) / (nu_expended + (self.eta) ** 2)

        R = scipy.special.digamma((nu_expended + 1) / 2)

        S = (self.lamb[np.newaxis, :, :] * self.eta * (self.eta**2 - 1)) / np.sqrt(
            (nu_expended + 1) * (nu_expended + (self.eta) ** 2) ** 3
        )

        T1_pdf = scipy.stats.t.pdf(M, nu_expended + 1)

        self.s4 = self.tik[:, np.newaxis, :] * (
            self.s1 - P - Q + R + S * (T1_pdf / T1_cdf)
        )

        return self

    def LL(self, x):
        """
        Calculate the log-likelihood of the model.

        Parameters
        ==========
        x : array-like
            Input parameter (not used in the calculation)

        Returns
        =======
        LL : float 
            Log-likelihood value
        """
        LL = np.sum(np.log(np.sum(self.p, axis=(1))), axis=0)

        return LL

    def update_lambda(self, y, X, k, j):
        """
        Calcul of the function h for the update of lambda.

        Parameters
        ==========
        y : float
            The input parameter.
        X : array-like
            The input data.
        k : int
            The cluster index.
        j : int
            The feature index.

        Returns
        =======
        result : float
            The result of the function h.
        """

        delta = y / np.sqrt(1 + y**2)

        term1 = delta * (1 - delta**2) * np.sum(self.tik[:, k], axis=0)

        diff_X_mu = X[:, j] - self.mu[j, k]
        term2_part1 = np.sum(
            (self.s1[:, j, k] * diff_X_mu**2) / self.sig[j, k] ** 2, axis=0
        )
        term2_part2 = np.sum(self.s3[:, j, k] / self.sig[j, k] ** 2, axis=0)
        term2 = delta * (term2_part1 + term2_part2)

        term3 = (1 + delta**2) * np.sum(
            (self.s2[:, j, k] * diff_X_mu) / self.sig[j, k] ** 2, axis=0
        )

        return term1 - term2 + term3

    def update_nu(self, y, k, j):
        """
        Calcul of the function i for the update of nu.

        Parameters
        ==========
        y : float
            The input parameter.

        k : int
            The cluster index.
        
        j : int
            The feature index.

        Returns
        =======
        result : float
            The result of the function i.
        """

        term1 = np.log(y / 2)

        term2 = scipy.special.digamma(y / 2)

        sum_tik_k = np.sum(self.tik[:, k], axis=0)

        term3 = np.sum(self.s4[:, j, k] - self.s1[:, j, k], axis=0) / sum_tik_k

        return term1 - term2 + term3

    def M_step(self, X):
        """
        Calcul new parameters of the model.

        Parameters
        ==========
        X : array-like
            The input data.
        """
        from joblib import Parallel, delayed
        import warnings

        # Update of alpha PX_CM-step-1
        self.alpha[:] = np.sum(self.tik[:, :], axis=0) / X.shape[0]

        self.mu = (
            np.sum(self.s1 * X[:, :, np.newaxis], axis=0)
            - self.delta * np.sum(self.s2, axis=0)
        ) / np.sum(self.s1, axis=0)

        # Update of sigma PX_CM-step-3
        self.sig[:, :] = np.sqrt(
            (
                np.sum(self.s1 * (X[:, :, np.newaxis] - self.mu[:, :]) ** 2, axis=0)
                - 2
                * self.delta
                * np.sum(self.s2 * (X[:, :, np.newaxis] - self.mu[:, :]), axis=0)
                + np.sum(self.s3, axis=0)
            )
            / (2 * (1 - self.delta**2) * np.sum(self.s1, axis=0))
        )

        # Update of lambda PX_CM-step-4
        def find_root_lamb(dim, k):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="invalid value encountered in log"
                )
                sol_lamb_1 = scipy.optimize.root(
                    self.update_lambda,
                    x0=self.lamb[dim, k],
                    args=(X, k, dim),
                    tol=10e-6,
                )
            return sol_lamb_1.x[0]

        def parallel_find_root_lamb(X):
            indices = np.ndindex(X.shape[1], self.n_cluster)
            results = Parallel(n_jobs=-1)(
                delayed(find_root_lamb)(*index) for index in indices
            )
            return np.array(results).reshape(X.shape[1], self.n_cluster)

        self.lamb = parallel_find_root_lamb(X)

        # Update of nu PX_CM-step-6
        def find_root_nu(dim, k):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="invalid value encountered in log"
                )
                sol_nu = scipy.optimize.root(
                    self.update_nu, x0=self.nu[dim, k], args=(k, dim), tol=10e-6
                )
            return sol_nu.x[0]

        def parallel_find_root_nu(X):
            indices = np.ndindex(X.shape[1], self.n_cluster)
            results = Parallel(n_jobs=-1)(
                delayed(find_root_nu)(*index) for index in indices
            )
            return np.array(results).reshape(X.shape[1], self.n_cluster)

        self.nu = parallel_find_root_nu(X)

        return self

    def predict_proba(self, X):
        """
        Predict the posterior probabilities of the data belonging to each cluster.

        Parameters
        ==========
        X : array-like
            The input data.

        Returns
        =======
        proba : array-like
            The posterior probabilities.
        """
        # Implementation of the predict_proba method
        p = self.phi(X)

        self.tik = p / np.sum(p, axis=(1))[:, np.newaxis]

        return self.tik

    def predict(self, X):
        """
        Predict the cluster labels for the data.

        Parameters
        ==========
        X : array-like
            The input data.

        Returns
        =======
        labels : array-like
            The predicted cluster labels.
        """
        # Implementation of the predict method
        proba = self.predict_proba(X)
        labels = []
        for i in range(proba.shape[0]):
            labels.append(np.argmax(proba[i, :]))
        return labels

    def confusion_matrix(self, y_true, y_pred=None):
        """
        Calculate the confusion matrix.

        Parameters
        ==========
        y_true : array-like
            The true labels.

        y_pred : array-like, default=None
            The predicted labels.

        Returns
        =======
        matrix : DataFrame
            The confusion matrix.
        """
        import pandas as pd

        if y_true is None:
            raise ValueError("Error: The true labels are missing, please provide them.")

        y_true = y_true.astype(int)

        # Predict the labels if they are not provided
        if y_pred is None:
            y_pred = self.predict(y_true).astype(int)
        else:
            y_pred = np.array(y_pred).astype(int)

        # Get the unique classes
        classes = np.unique(np.concatenate((y_true, y_pred)))

        # Create the confusion matrix
        matrix = pd.DataFrame(0, index=classes, columns=classes)

        # Count the occurrences of each class
        for i in range(len(y_true)):
            true_label = y_true[i]
            pred_label = y_pred[i]
            matrix.at[true_label, pred_label] += 1

        return matrix

    def ARI(self, x, y):
        """
        Compute the accuracy of the model.

        Parameters
        ==========
        x : array-like
            The input data.

        y : array-like
            The true labels.

        Returns
        =======
        acc : float
            The accuracy.
        """
        from sklearn.metrics import adjusted_rand_score

        ari = adjusted_rand_score(y, np.argmax(self.predict_proba(x), axis=1))

        return print("ARI:", ari)

    def save(self, filename: str):
        """
        Save the model to a file.

        Parameters
        ==========
        filename : str
            The name of the file.
        """
        import h5py

        if not filename.endswith(".h5"):
            filename = f"{filename}.h5"

        if not filename:
            raise ValueError(
                "Error: The filename is empty, please provide a valid filename."
            )

        if not os.path.exists("Models_folder"):
            os.makedirs("Models_folder")

        with h5py.File(f"Models_folder/{filename}", "w") as f:
            f.create_dataset("mu", data=self.mu)
            f.create_dataset("sig", data=self.sig)
            f.create_dataset("nu", data=self.nu)
            f.create_dataset("lamb", data=self.lamb)
            f.create_dataset("alpha", data=self.alpha)
            f.create_dataset("E_log_likelihood", data=self.E_log_likelihood)

    def load(self, filename: str):
        """
        Load matrices from a given file.

        Parameters
        ==========
        filename : str
            The path to the file containing the matrices.
        """
        import h5py

        with h5py.File(filename, "r") as f:
            self.mu = f["mu"][:]
            self.sig = f["sig"][:]
            self.nu = f["nu"][:]
            self.lamb = f["lamb"][:]
            self.alpha = f["alpha"][:]
            self.E_log_likelihood = f["E_log_likelihood"][:]


class SkewTUniformMixture:
    """
    Skew-t Unifrom Mixture Model for clustering with uniform background using PX-EM algorithm.

    Parameters
    ==========
    n_cluster : int
        The number of mixture components (clusters). The cluster uniform is not included in this number.

    n_iter : int, default=10
        The number of iterations to perform during the parameter estimation.

    tol : float, default=1e-5
        The convergence threshold. Iterations will stop when the
        improvement is below this threshold.

    init : {'random', 'kmeans', 'gmm', 'params'}, default='random'
        The method used to initialize the parameters.
        Must be one of:

        - 'random': Parameters are initialized randomly.
        - 'params': User-provided parameters are used for initialization.
        - 'kmeans': Parameters are initialized using K-means.
        - 'gmm': Parameters are initialized using a Gaussian Mixture Model.

    params : dict, default=None
        The user-provided initial parameters. Used only if `init` is 'params'.

    References
    ==========

    [1] `Lin, Tsung & Lee, Jack & Hsieh, Wan. (2007). Robust mixture models using the skew-t distribution. Statistics and Computing. 17. 81-92. 10.1007/s11222-006-9005-8. <https://doi.org/10.1007/s11222-006-9005-8>`_

    Notes
    =====

    For more information, refer to the documentation:ref:`doc.mixture.SkewTUniformMixture`

    Examples
    ========
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
    """

    def __init__(
        self, n_cluster: int, n_iter=10, tol=1e-8, init="gmm", params=None, n_init_gmm=8
    ):
        self.n_cluster = n_cluster
        self.n_iter = n_iter
        self.tol = tol
        self.init_method = init
        if self.init_method == "params":
            self.params = params

        if self.init_method == "gmm":
            self.n_init_gmm = n_init_gmm

    def fit(self, X):
        """
        Fits the SkewMM model to the input data.

        Parameters
        ==========
        X : array-like 
            Shape (n_samples, n_features). Each sample is represented by a feature vector Xi = [e_i, i_i, H_i, a_i].
        """

        # Implementation of the fit method
        if self.init_method == "random":
            best_params = self.initialisation_random(X)
            self.initialisation_params(best_params, X)

        elif self.init_method == "kmeans":
            best_params = self.initialisation_kmeans(X)
            self.initialisation_params(best_params, X)

        elif self.init_method == "params":
            self.initialisation_params(self.params, X)

        elif self.init_method == "gmm":
            best_params = None
            best_LL = -np.inf

            params_to_test = [
                self.initialisation_gmm(X) for _ in range(self.n_init_gmm)
            ]

            # Test each set of parameters and keep the best one
            for params in params_to_test:
                self.initialisation_params(params, X)
                self.p = self.phi(X)
                LL = self.LL(X)
                if LL > best_LL:
                    best_LL = LL
                    best_params = params

            # Apply the best parameters found
            self.initialisation_params(best_params, X)

        elif self.init_method == "likelihood":
            param_random = self.initialisation_random(X)
            LLN_random = self.E_step(X)
            param_gmm = self.initialisation_gmm(X)
            LLN_gmm = self.E_step(X)

            if LLN_random > LLN_gmm:
                self.initialisation_params(param_random, X)
                print("initialization random")
            else:
                self.initialisation_params(param_gmm, X)
                print("initialization gmm")
        else:
            raise ValueError(
                f"Error: The initialization method {self.init_method} is not recognized, please choose from 'random', 'kmeans', 'params', 'gmm' ou 'likelihood'"
            )

        print("initialization method :", self.init_method)

        self.E_log_likelihood = [-np.inf]

        self.save("track_0")

        i = 0

        while i < self.n_iter + 1:
            print(f"iteration: {i}/{self.n_iter}")

            self.E_step(X)
            E_log_likelihood_new = self.LL(X)

            if np.abs(E_log_likelihood_new - self.E_log_likelihood[i]) <= self.tol:
                if i < 2:
                    print("reinitialization not enough iteration")
                    return self.fit(X)
                else:
                    break

            if E_log_likelihood_new < self.E_log_likelihood[i]:
                if i < 2:
                    print("reinitialization not enough iteration")
                    return self.fit(X)
                else:
                    break

            self.E_log_likelihood.append(E_log_likelihood_new)
            self.M_step(X)

            if np.any(np.isnan(self.sig)):
                if i < 2:
                    print("reinitialization : sig nan")
                    return self.fit(X)
                else:
                    self.load(f"Models_folder/track_{i-1}.h5")
                    self.n_iter = i - 1
                    break

            if np.any(self.nu < 0):
                if i < 2:
                    print("reinitialization : nu <0")
                    return self.fit(X)
                else:
                    self.load(f"Models_folder/track_{i-1}.h5")
                    self.n_iter = i - 1
                    break

            if np.any(np.diagonal(self.sig, axis1=0, axis2=1) < 0):
                if i < 2:
                    np.diagonal(self.sig, axis1=0, axis2=1)

                    print("sig", self.sig)

                    print("eig:", np.diagonal(self.sig, axis1=0, axis2=1))
                    print("reinitialization : negative equity")
                    return self.fit(X)
                else:
                    model = self.load(f"Models_folder/track_{i-1}.h5")
                    self.n_iter = i - 1
                    break

            self.save(f"track_{i}")

            i += 1

        return self

    def initialisation_random(self, X):
        """
        Random initialization method for the SkewMM algorithm.

        Parameters
        ==========
        X : array-like
            Input data array.

        Returns
        =======
        dict : dict
            A dictionary containing the initialized parameters:

            - 'mu': numpy array
                Matrix of means.

            - 'sig': numpy array
                Matrix of covariances.

            - 'nu': numpy array
                Matrix of degrees of freedom.

            - 'lamb': numpy array
                Matrix of skewness parameters.

            - 'alpha': numpy array
                Array of cluster proportions.
        """
        # Implementation of the random initialization method
        max_x = np.max(X, axis=0)[:, np.newaxis] if X.ndim > 1 else np.max(X)

        # initialization of the average matrix
        mu = np.random.rand(X.shape[1], self.n_cluster) * max_x

        # initialization of the covariance matrix
        sig = np.ones((X.shape[1], self.n_cluster))

        # initialization of the degree of freedom
        nu = np.random.rand(X.shape[1], self.n_cluster)

        # initialization of the skewness parameter
        lamb = np.random.uniform(low=1e-6, high=5.0, size=(X.shape[1], self.n_cluster))

        # initialization of the prior, proportion of data in cluster k
        alpha = np.random.rand(self.n_cluster)

        return {"mu": mu, "sig": sig, "nu": nu, "lamb": lamb, "alpha": alpha}

    def initialisation_params(self, params, X):
        """
        Initialize the parameters of the SkewMM model.

        Parameters
        ==========
        params : dict
            A dictionary containing the initial values for the model parameters.

            - 'mu' : ndarray
                The mean vectors for each cluster. Shape: (n_features, n_cluster).

            - 'sig' : ndarray
                The covariance matrices for each cluster. Shape: (n_features, n_cluster).

            - 'nu' : ndarray
                The degrees of freedom for each cluster. Shape: (n_features, n_cluster).

            - 'lamb' : ndarray
                The skewness parameters for each cluster. Shape: (n_features, n_cluster).

            - 'alpha' : ndarray
                The mixing proportions for each cluster. Shape: (n_cluster,).

        X : ndarray
            The input data. Shape: (n_samples, n_features).
        """
        if params["mu"].shape != (X.shape[1], self.n_cluster):
            raise ValueError(
                f"Error: The size of the matrix must be {(X.shape[1], self.n_cluster)}, but it is {params['mu'].shape}"
            )
        self.mu = np.array(params["mu"], dtype=float)

        if params["sig"].shape != (X.shape[1], self.n_cluster):
            raise ValueError(
                f"Error: The size of the matrix must be {(X.shape[1], self.n_cluster)}, but it is {params['sig'].shape}"
            )
        self.sig = np.array(params["sig"], dtype=float)

        if params["nu"].shape != (X.shape[1], self.n_cluster):
            raise ValueError(
                f"Error: The size of the matrix must be {(X.shape[1], self.n_cluster)}, but it is {params['nu'].shape}"
            )
        self.nu = np.array(params["nu"], dtype=float)

        if params["lamb"].shape != (X.shape[1], self.n_cluster):
            raise ValueError(
                f"Error: The size of the matrix must be {(X.shape[1], self.n_cluster)}, but it is {params['lamb'].shape}"
            )
        self.lamb = np.array(params["lamb"], dtype=float)

        if params["alpha"].shape != (self.n_cluster,):
            raise ValueError(
                f"Error: The size of the matrix must be {(self.n_cluster)}, but it is {params['alpha'].shape}"
            )
        self.alpha = np.array(params["alpha"], dtype=float)

        # Add the noise cluster
        self.alpha = np.append(self.alpha, 0.9)
        self.alpha = self.alpha/ np.sum(self.alpha)

        return self

    def initialisation_kmeans(self, X, default_n_init="auto"):
        """
        Initializes the parameters for the SkewMM algorithm using the K-means initialization method.

        Parameters
        ==========
        X : ndarray
            The input data matrix of shape (n_samples, n_features).

        default_n_init : int, default='auto'
            The number of times the K-means algorithm will be run with different centroid seeds. Default is 'auto'.

        Returns
        =======
        dict : dict
            A dictionary containing the initialized parameters:

            - 'mu': numpy array
                Matrix of means.

            - 'sig': numpy array
                Matrix of covariances.

            - 'nu': numpy array
                Matrix of degrees of freedom.

            - 'lamb': numpy array
                Matrix of skewness parameters.

            - 'alpha': numpy array
                Array of cluster proportions.
        """
        from sklearn.cluster import KMeans

        # Implementation of the K-means initialization method
        # Use KMeans to get the cluster centers
        kmeans = KMeans(
            n_clusters=self.n_cluster, n_init=20
        )  # pas de cluster pour le bruit
        kmeans.fit(X)
        cluster_centers = kmeans.cluster_centers_

        # Initializations of the mean matrix
        mu = cluster_centers.T

        # Initializations of the covariance matrix
        sig = np.ones((X.shape[1], self.n_cluster))

        # Initializations of the degree of freedom
        nu = np.random.rand(X.shape[1], self.n_cluster)

        # Initializations of the skewness parameter
        lamb = np.random.uniform(low=1e-6, high=10.0, size=(X.shape[1], self.n_cluster))

        # Initializations of the prior, proportion of data in cluster k
        alpha = np.random.rand(self.n_cluster)

        return {"mu": mu, "sig": sig, "nu": nu, "lamb": lamb, "alpha": alpha}

    def initialisation_gmm(self, X):
        """
        Initialize the parameters for the Gaussian Mixture Model (GMM).

        Parameters
        ==========
        X : array-like
            Input data matrix of shape (n_samples, n_features).

        Returns
        =======
        dict : dict
            A dictionary containing the initialized parameters:

            - 'mu': numpy array
                Matrix of means.

            - 'sig': numpy array
                Matrix of covariances.

            - 'nu': numpy array
                Matrix of degrees of freedom.

            - 'lamb': numpy array
                Matrix of skewness parameters.

            - 'alpha': numpy array
                Array of cluster proportions.
        """
        from sklearn.mixture import GaussianMixture

        # Use the Gaussian Mixture Model to initialize the parameters
        gmm = GaussianMixture(
            n_components=self.n_cluster,
            covariance_type="diag",
            reg_covar=1e-6,
            tol=1e-6,
        )
        gmm.fit(X)

        # Initializations of the mean matrix with the means of the GMM
        mu = gmm.means_.T

        # Initializations of the covariance matrix with the covariances of the GMM
        sig = gmm.covariances_.T

        # Initialisations of the degree of freedom
        nu = np.random.uniform(0., 10., size=(X.shape[1], self.n_cluster))

        # Initialisations of the skewness parameter
        lamb = np.random.uniform(low=1e-6, high=1e-5, size=(X.shape[1], self.n_cluster))

        # Initializations of the prior, proportion of data in cluster k
        alpha = gmm.weights_

        return {"mu": mu, "sig": sig, "nu": nu, "lamb": lamb, "alpha": alpha}

    def g_of_nu(self, x, nu):
        """
        Calculate the value of g(nu) for a given x and nu.

        Parameters
        ==========
        x : float
            The input value.
        
        nu : float
            The parameter value.
        """
        A = scipy.special.digamma((nu + 2) / 2)
        B = -scipy.special.digamma((nu + 1) / 2)
        C = -np.log(1 + (x**2) / (nu + 1))
        D = ((nu + 1) * x**2 - nu - 1) / ((nu + 1) * (nu + 1 + x**2))
        return A + B + C + D

    def phi(self, x):
        """
        Calculates the probability density function (PDF) for each data point in x.

        Parameters
        ==========
        x : array-like
            Shape : (n_samples, n_features). The input data points.

        Returns
        =======
        p : array-like 
            Shape : (n_samples, n_cluster). The PDF values for each data point in x.
        """
        p = np.zeros((x.shape[0], self.n_cluster+1))

        for index, value in enumerate(x):
            p[index, :-1] = self.alpha[:-1]
            p[index, -1] = self.alpha[-1]
            for dim in range(x.shape[1]):
                p[index, :-1] *= Skew.pdf(
                    x = value[dim],
                    mu = self.mu[dim, :],
                    sigma = self.sig[dim, :],
                    nu = self.nu[dim, :],
                    lamb = self.lamb[dim, :]
                )

                p[index, -1] *= 1 / (x[:, dim].max() - x[:, dim].min())         

        return p

    def E_step(self, X):
        """
        Performs the E-step of the SkewMM algorithm.
        """

        # Implementation of the predict_proba method
        self.p = self.phi(X)

        self.tik = self.p / np.sum(self.p, axis=1)[:, np.newaxis]

        # Calculation of s1
        nu_expended = self.nu[np.newaxis, :, :]

        self.eta = (X[:, :, np.newaxis] - self.mu[np.newaxis, :, :]) / self.sig[
            np.newaxis, :, :
        ]

        M = (
            self.lamb[np.newaxis, :, :]
            * self.eta
            * np.sqrt((self.nu[np.newaxis, :, :] + 1) / (nu_expended + (self.eta) ** 2))
        )
        T3_cdf = scipy.stats.t.cdf(
            M * np.sqrt((nu_expended + 3) / (nu_expended + 1)), df=(nu_expended + 3)
        )
        T1_cdf = scipy.stats.t.cdf(M, df=(nu_expended + 1))

        self.s1 = (
            self.tik[:, np.newaxis, :-1]
            * ((nu_expended + 1) / (nu_expended + (self.eta) ** 2))
            * (T3_cdf / T1_cdf)
        )

        # Calculation of s2
        self.delta = self.lamb / np.sqrt(1 + self.lamb**2)
        term1 = (
            self.delta[np.newaxis, :, :]
            * (X[:, :, np.newaxis] - self.mu[np.newaxis, :, :])
            * self.s1
        )

        f = np.zeros((X.shape[0], X.shape[1], self.n_cluster))

        for dim in range(X.shape[1]):
            f[:, dim, :] = Skew.pdf(
                X[:, dim][:, np.newaxis],
                self.mu[dim, :],
                self.sig[dim, :],
                self.nu[dim, :],
                self.lamb[dim, :],
            )

        term2 = np.sqrt(1 - self.delta[np.newaxis, :, :] ** 2) / (np.pi * f[:, :, :])

        term3 = (self.eta**2) / (
            nu_expended * (1 - (self.delta[np.newaxis, :, :]) ** 2)
        )
        term4 = (term3 + 1) ** (-((nu_expended / 2) + 1))
        self.s2 = term1 + self.tik[:, np.newaxis, :-1] * term2 * term4

        # Calculation of s3
        partie1 = (
            self.delta[np.newaxis, :, :] ** 2
            * (X[:, :, np.newaxis] - self.mu[np.newaxis, :, :]) ** 2
            * self.s1
        )

        partie2 = (1 - self.delta[np.newaxis, :, :] ** 2) * (
            self.sig[np.newaxis, :, :] ** 2
        )

        partie3 = (
            self.delta[np.newaxis, :, :]
            * (X[:, :, np.newaxis] - self.mu[np.newaxis, :, :])
            * np.sqrt(1 - self.delta[np.newaxis, :, :] ** 2)
        ) / (np.pi * f)

        partie4 = self.eta**2 / (
            nu_expended * (1 - (self.delta[np.newaxis, :, :]) ** 2)
        )
        partie5 = (partie4 + 1) ** (-((self.nu[np.newaxis, :, :] / 2) + 1))

        self.s3 = partie1 + self.tik[:, np.newaxis, :-1] * (partie2 + partie3 * partie5)

        # Calculation of s4
        P = np.log((self.eta**2 + nu_expended) / 2)

        Q = (nu_expended + 1) / (nu_expended + (self.eta) ** 2)

        R = scipy.special.digamma((nu_expended + 1) / 2)

        S = (self.lamb[np.newaxis, :, :] * self.eta * (self.eta**2 - 1)) / np.sqrt(
            (nu_expended + 1) * (nu_expended + (self.eta) ** 2) ** 3
        )

        T1_pdf = scipy.stats.t.pdf(M, nu_expended + 1)

        self.s4 = self.tik[:, np.newaxis, :-1] * (
            self.s1 - P - Q + R + S * (T1_pdf / T1_cdf)
        )

        return self

    def LL(self, x):
        """
        Calculate the log-likelihood of the model.

        Parameters
        ==========
        x : array-like
            Input parameter (not used in the calculation)

        Returns
        =======
        LL : float 
            Log-likelihood value
        """
        LL = np.sum(np.log(np.sum(self.p, axis=(1))), axis=0)

        return LL

    def update_lambda(self, y, X, k, j):
        """
        Calcul of the function h for the update of lambda.

        Parameters
        ==========
        y : float
            The input parameter.
        X : array-like
            The input data.
        k : int
            The cluster index.
        j : int
            The feature index.

        Returns
        =======
        result : float
            The result of the function h.
        """

        delta = y / np.sqrt(1 + y**2)

        term1 = delta * (1 - delta**2) * np.sum(self.tik[:, k], axis=0)

        diff_X_mu = X[:, j] - self.mu[j, k]
        term2_part1 = np.sum(
            (self.s1[:, j, k] * diff_X_mu**2) / self.sig[j, k] ** 2, axis=0
        )
        term2_part2 = np.sum(self.s3[:, j, k] / self.sig[j, k] ** 2, axis=0)
        term2 = delta * (term2_part1 + term2_part2)

        term3 = (1 + delta**2) * np.sum(
            (self.s2[:, j, k] * diff_X_mu) / self.sig[j, k] ** 2, axis=0
        )

        return term1 - term2 + term3

    def update_nu(self, y, k, j):
        """
        Calcul of the function i for the update of nu.

        Parameters
        ==========
        y : float
            The input parameter.

        k : int
            The cluster index.
        
        j : int
            The feature index.

        Returns
        =======
        result : float
            The result of the function i.
        """

        term1 = np.log(y / 2)

        term2 = scipy.special.digamma(y / 2)

        sum_tik_k = np.sum(self.tik[:, k], axis=0)

        term3 = np.sum(self.s4[:, j, k] - self.s1[:, j, k], axis=0) / sum_tik_k

        return term1 - term2 + term3

    def M_step(self, X):
        """
        Calcul new parameters of the model.

        Parameters
        ==========
        X : array-like
            The input data.
        """
        from joblib import Parallel, delayed
        import warnings

        # Update of alpha PX_CM-step-1
        self.alpha[:] = np.sum(self.tik[:, :], axis=0) / X.shape[0]

        self.mu = (
            np.sum(self.s1 * X[:, :, np.newaxis], axis=0)
            - self.delta * np.sum(self.s2, axis=0)
        ) / np.sum(self.s1, axis=0)

        # Update of sigma PX_CM-step-3
        self.sig[:, :] = np.sqrt(
            (
                np.sum(self.s1 * (X[:, :, np.newaxis] - self.mu[:, :]) ** 2, axis=0)
                - 2
                * self.delta
                * np.sum(self.s2 * (X[:, :, np.newaxis] - self.mu[:, :]), axis=0)
                + np.sum(self.s3, axis=0)
            )
            / (2 * (1 - self.delta**2) * np.sum(self.s1, axis=0))
        )

        # Update of lambda PX_CM-step-4
        def find_root_lamb(dim, k):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="invalid value encountered in log"
                )
                sol_lamb_1 = scipy.optimize.root(
                    self.update_lambda,
                    x0=self.lamb[dim, k],
                    args=(X, k, dim),
                    tol=10e-6,
                )
            return sol_lamb_1.x[0]

        def parallel_find_root_lamb(X):
            indices = np.ndindex(X.shape[1], self.n_cluster)
            results = Parallel(n_jobs=-1)(
                delayed(find_root_lamb)(*index) for index in indices
            )
            return np.array(results).reshape(X.shape[1], self.n_cluster)

        self.lamb = parallel_find_root_lamb(X)

        # Update of nu PX_CM-step-6
        def find_root_nu(dim, k):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="invalid value encountered in log"
                )
                sol_nu = scipy.optimize.root(
                    self.update_nu, x0=self.nu[dim, k], args=(k, dim), tol=10e-6
                )
            return sol_nu.x[0]

        def parallel_find_root_nu(X):
            indices = np.ndindex(X.shape[1], self.n_cluster)
            results = Parallel(n_jobs=-1)(
                delayed(find_root_nu)(*index) for index in indices
            )
            return np.array(results).reshape(X.shape[1], self.n_cluster)

        self.nu = parallel_find_root_nu(X)

        return self

    def predict_proba(self, X):
        """
        Predict the posterior probabilities of the data belonging to each cluster.

        Parameters
        ==========
        X : array-like
            The input data.

        Returns
        =======
        proba : array-like
            The posterior probabilities.
        """
        # Implementation of the predict_proba method
        p = self.phi(X)

        self.tik = p / np.sum(p, axis=(1))[:, np.newaxis]

        return self.tik

    def predict(self, X):
        """
        Predict the cluster labels for the data.

        Parameters
        ==========
        X : array-like
            The input data.

        Returns
        =======
        labels : array-like
            The predicted cluster labels.
        """
        # Implementation of the predict method
        proba = self.predict_proba(X)
        labels = []
        for i in range(proba.shape[0]):
            labels.append(np.argmax(proba[i, :]))
        return labels

    def confusion_matrix(self, y_true, y_pred):
        """
        Calculate the confusion matrix.

        Parameters
        ==========
        y_true : array-like
            The true labels.

        y_pred : array-like
            The predicted labels.

        Returns
        =======
        matrix : DataFrame
            The confusion matrix.
        """
        import pandas as pd

        if y_true is None:
            raise ValueError("Error: The true labels are missing, please provide them.")

        y_true = y_true.astype(int)

        y_pred = np.array(y_pred).astype(int)

        # Get the unique classes
        classes = np.unique(np.concatenate((y_true, y_pred)))

        # Create the confusion matrix
        matrix = pd.DataFrame(0, index=classes, columns=classes)

        # Count the occurrences of each class
        for i in range(len(y_true)):
            true_label = y_true[i]
            pred_label = y_pred[i]
            matrix.at[true_label, pred_label] += 1

        return matrix

    def ARI(self, x, y):
        """
        Compute the accuracy of the model.

        Parameters
        ==========
        x : array-like
            The input data.

        y : array-like
            The true labels.

        Returns
        =======
        acc : float
            The accuracy.
        """
        from sklearn.metrics import adjusted_rand_score

        ari = adjusted_rand_score(y, np.argmax(self.predict_proba(x), axis=1))

        return print("ARI:", ari)

    def save(self, filename: str):
        """
        Save the model to a file.

        Parameters
        ==========
        filename : str
            The name of the file.
        """
        import h5py

        if not filename.endswith(".h5"):
            filename = f"{filename}.h5"

        if not filename:
            raise ValueError(
                "Error: The filename is empty, please provide a valid filename."
            )

        if not os.path.exists("Models_folder"):
            os.makedirs("Models_folder")

        with h5py.File(f"Models_folder/{filename}", "w") as f:
            f.create_dataset("mu", data=self.mu)
            f.create_dataset("sig", data=self.sig)
            f.create_dataset("nu", data=self.nu)
            f.create_dataset("lamb", data=self.lamb)
            f.create_dataset("alpha", data=self.alpha)
            f.create_dataset("E_log_likelihood", data=self.E_log_likelihood)

    def load(self, filename: str):
        """
        Load matrices from a given file.

        Parameters
        ==========
        filename : str
            The path to the file containing the matrices.
        """
        import h5py

        with h5py.File(filename, "r") as f:
            self.mu = f["mu"][:]
            self.sig = f["sig"][:]
            self.nu = f["nu"][:]
            self.lamb = f["lamb"][:]
            self.alpha = f["alpha"][:]
            self.E_log_likelihood = f["E_log_likelihood"][:]


