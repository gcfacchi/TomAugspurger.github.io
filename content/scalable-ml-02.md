---
title: "Scalable Machine Learning (interlude)"
date: 2017-09-15
slug: scalable-ml-interlude-01
---

*This work is supported by [Anaconda Inc.] and the Data Driven Discovery
Initiative from the [Moore Foundation].*

This is a bit of an interlude in my series on scalable machine learning, but I
wanted to collect feedback on an idea I had today. It's less of a "how to" and
more of a "thoughts?".

Scikit-learn supports out-of-core learning (fitting a model on a dataset that
doesn't fit in RAM), through it's `partial_fit` API. See
[here](http://scikit-learn.org/stable/modules/scaling_strategies.html#scaling-with-instances-using-out-of-core-learning).

The basic idea is that, *for certain estimators*, learning can be done in
batches. The estimator will see a batch, and then incrementally update whatever
it's learning (the coefficients, for example).

Unfortunately, the `partial_fit` API doesn't play that nicely with my favorite
part of scikit-learn:
[pipelines](http://scikit-learn.org/stable/modules/pipeline.html#pipeline). You
would essentially need every chain in the pipeline to have an out-of-core
`parital_fit` version, which isn't really feasible. Setting that aside, it
wouldn't be great for a user, since working with generators of datasets is
awkward.

Fortunately, we *have* a great data containers for larger than memory arrays and
dataframes: `dask.array` and `dask.dataframe`. We can

1. Use dask for pre-processing data in an out-of-core manner
2. Use scikit-learn to fit the actual model, out-of-core, using the
   `partial_fit` API

And all of this can be done in a pipeline. The rest of this post shows how.


```python
from daskml.datasets import make_classification
from daskml.linear_model import BigSGDClassifier

from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
```

`daskml` is a prototype library where I'm collection these thoughts. It's not
ready for production (yet).

Let's make an `X` and `y` for classification.


```python
X, y = make_classification(n_samples=10_000, chunks=1_000)
```

These are dask arrays:


```python
X
```




    dask.array<array, shape=(10000, 20), dtype=float64, chunksize=(1000, 20)>




```python
y
```




    dask.array<array, shape=(10000,), dtype=int64, chunksize=(1000,)>



To demonstrate the idea, we'll have a small pipeline

1. Scale the features by mean and variance
2. Fit an `SGDClassifer`

Since scikit-learn isn't dask-aware, we'll write our own `StandardScaler`. This
isn't too many lines of code:


```python
class StandardScaler(TransformerMixin):

    def fit(self, X, y=None):
        self.mean_ = X.mean(0)
        self.var_ = X.var(0)
        return self

    def transform(self, X, y=None):
        return (X - self.mean_) / self.var_
```

And now for the pipeline:


```python
pipe = make_pipeline(
    StandardScaler(),
    BigSGDClassifier(classes=[0, 1], max_iter=1000, tol=1e-3, random_state=2),
)

pipe.fit(X, y)
```


    Pipeline(memory=None,
         steps=[('standardscaler', <__main__.StandardScaler object at 0x1137efa20>),
                ('bigsgdclassifier', BigSGDClassifier(
                    alpha=0.0001, average=False, class_weight=None,
                    classes=[0, 1], epsilon=0.1, eta0=0.0, fit_intercept=True,
                    l1_ratio=0.15, learning_rate='optimal', loss='hinge',
                    max_iter=1000, n_iter=None, n_jobs=1, penalty='l2', power_t=0.5,
                    random_state=2, shuffle=True, tol=0.001, verbose=0,
                    warm_start=False))])

```python
pipe.steps[-1][1].coef_
```

    array([[ -3.84269726,  -0.37780912,  15.68333536,  -1.31574306,
              2.70781476,  -2.55583381,  -0.39162044,  -1.18764602,
              1.74171432,   0.38190678,  -1.95610583,   5.51880009,
            -10.83082117,  -1.18993518,   1.56220091,   0.65688068,
            14.64347836,   0.76979726,   1.11516644,  -2.99760032]])


Somewhat anticlimatic, I'll admit, but I'm excited about it! We get to write
NumPy-like code, operate on larger datsets, and use (some) scikit-learn
estimators, without modify scikit-learn at all!

## How?

The implementation is equally exiting to me. It essentially comes down to two
methods


```python
import dask.array as da

def _as_blocks(X, y):
    X_blocks = X.to_delayed().flatten().tolist()
    y_blocks = y.to_delayed().flatten().tolist()
    return zip(X_blocks, y_blocks)


class _BigDataMixin:

    def fit(self, X, y=None):
        pairs = _as_blocks(X, y)
        P = X.shape[1]

        for i, (xx, yy) in enumerate(pairs):
            xx = da.from_delayed(xx, shape=(X.chunks[0][i], P), dtype=X.dtype)
            yy = da.from_delayed(yy, shape=(y.chunks[0][i],), dtype=y.dtype)
            self.partial_fit(xx, yy, classes=self.classes)
```

`_as_blocks` is a little helper for going from a `dask.array` to a list of `dask.Delayed` objects.


```python
xx, yy = next(_as_blocks(X, y))
xx, yy
```


    (Delayed(('array-faee43bf99d08c2158b52f3c76cccfdf', 0, 0)),
     Delayed(('array-0e5a80aa67c1727b5b452eb624c3b30e', 0)))


In the `_BigDataMixin` class (I chuckle every time at that name), we iterate over each of the `xx, yy` pairs. Wd do a `da.from_delayed` dance to get a (small) `dask.Array` that looks array-like enough for NumPy to operate on it. That's what's passed into the `partial_fit` of the class it's mixed in with.

It's important to stress that we don't get any parallelism here. This is entirely sequential. This is more about the user-convenience of getting to use dask arrays for exploring larger-than-memory datasets. We get to use a complex scikit-learn estimator on a `dask.array` in ~20 lines of code. I'll take it for now.

Anyway, let me know what you think. I'm pretty excited about this because it removes some of the friction around using sckit-learn Pipelines with the out-of-core estimators. I'll be packaging this up in `daskml` to make it more usable for the community, but wanted to get some feedback on the idea first.

You can download a notebook demonstrating this [here](http://nbviewer.jupyter.org/gist/TomAugspurger/6306a5eb7389351164801fcbf2945521).
