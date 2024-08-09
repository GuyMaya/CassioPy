import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Grid_skewt:
    """
    A class representing a grid of SkewT mixture models.

    Parameters
    ==========
    n_cluster : array-like
        The number of clusters in the mixture model.

    n_fits : int
        The number of fits to perform for each number of clusters.

    n_iter : int
        The number of iterations to perform for each fit.

    mixture : str
        The type of mixture model to use. Choose between :
        
        - 'SkewTUniformMixture'
        - 'SkewTMixture'

    init : {'random', 'kmeans', 'gmm', 'params'}, default='gmm'
        The method used to initialize the parameters.
        Must be one of:

        - 'random': Parameters are initialized randomly.
        - 'params': User-provided parameters are used for initialization.
        - 'kmeans': Parameters are initialized using K-means.
        - 'gmm': Parameters are initialized using a Gaussian Mixture Model.

    Attributes
    ==========

    bic : array-like
        The BIC scores. Shape (n_cluster, n_fits).
        
    Examples
    ========
    >>> from cassiopy.grid import Grid_skewt
    >>> grid = Grid_skewt(n_cluster=range(1,3,1), n_fits=2, n_iter=10, mixture='SkewTMixture')
    >>> grid.fit(x)
    >>> grid.bic
    array([[56.,  3.1],
        [ 5.5,  16.2]])
    >>> BM =  grid.best_model()
    >>> BM.predict(x)
    """

    def __init__(self, n_cluster=5, n_fits=2, n_iter=10, mixture=None, init='gmm', verbose=0):
        self.n_cluster = n_cluster
        self.n_fits = n_fits
        self.n_iter = n_iter
        self.verbose = verbose
        self.init = init

        if mixture is None:
            raise ValueError(
                "mixture model must be specified, please choose between 'SkewTUniformMixture' or 'SkewTMixture'"
            )
        if mixture == "SkewTUniformMixture":
            from cassiopy.mixture import SkewTUniformMixture

            self.mixture = SkewTUniformMixture
        elif mixture == "SkewTMixture":
            from cassiopy.mixture import SkewTMixture

            self.mixture = SkewTMixture

    def fit(self, x):
        """
        Fits the mixture model to the given data.

        Parameters
        ==========
        x : array-like
            The input data.

        Returns
        =======
        models : list
            A list of fitted mixture models.
        """
        self.bic = np.zeros((len(self.n_cluster), self.n_fits))
        self.models = []
        # loop through each number of Gaussians and compute the BIC, and save the model
        for i, j in zip(range(len(self.n_cluster)), self.n_cluster):
            # create mixture model with j components
            model = self.mixture(n_cluster=j, n_iter=self.n_iter, init=self.init, verbose=self.verbose)
            for k in range(self.n_fits):
                model.fit(x)
                self.bic[i, k] = model.bic(x)

                self.models.append(model)

        return self.models

    def best_model(self):
        """
        Returns the best model based on the BIC score.

        Returns
        =======
        best_model : object
            The best fitted mixture model.

        Notes
        =====

        For more information, refer to the documentation :ref:`doc.mixture.BIC`
        """

        f = np.where(self.bic == self.bic.min())[1][0]
        k = np.where(self.bic == self.bic.min())[0][0]

        best_model = self.models[f + k * self.n_fits]

        return best_model

    def best_nbre_cluster(self):
        """
        Returns the number of clusters for the best model.

        Returns
        =======
        nbre_cluster : int
            The number of clusters for the best model.
        """

        nbre_cluster = self.n_cluster[np.where(self.bic == self.bic.min())[0][0]]
        return nbre_cluster

    def plot_bic(self):
        """
        Plots the BIC scores in function of the numbers of clusters.
        """
        sns.lineplot(
            x=self.n_cluster, y=self.bic[:, np.where(self.bic == self.bic.min())[1][0]]
        )
        # Ajout des noms d'axes
        plt.xlabel("Nbre clusters")
        plt.ylabel("BIC")
        plt.show()

