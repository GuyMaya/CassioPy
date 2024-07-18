## GenePy - 

Skew Mixture Model for clustering

```python
>>> from genepi.skew_mixture import SkewMixtureModel
>>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
>>> model = SkewMixtureModel(n_cluster=2, n_iter=10, init='gmm')
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
```

Skew -t distribution

```python
>>> from genepi.stats import SkewSet
>>> skew_set = SkewSet()
>>> data, y_true = skew_set.generate(n_samples=100, n_dim=1, n_cluster=4, random_state=42)
```


## Installation

Install from Pypi using `pip` :

    $ pip install skew_mixture

## Documentation

