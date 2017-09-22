---
title: "Scalable Machine Learning (Part 2): Partial Fit"
date: 2017-09-15
slug: scalable-ml-interlude-01
---

*This work is supported by [Anaconda, Inc.](https://www.anaconda.com/) and the
Data Driven Discovery Initiative from the [Moore Foundation](https://www.moore.org/).*

This is part two of my series on scalable machine learning.

- [Small Fit, Big Predict](scalable-ml-01)
- [Scikit-Learn Partial Fit](scalable-ml-02)

You can download a notebook of this post [here][notebook].

---

Scikit-learn supports out-of-core learning (fitting a model on a dataset that
doesn't fit in RAM), through it's `partial_fit` API. See
[here](http://scikit-learn.org/stable/modules/scaling_strategies.html#scaling-with-instances-using-out-of-core-learning).

The basic idea is that, *for certain estimators*, learning can be done in
batches. The estimator will see a batch, and then incrementally update whatever
it's learning (the coefficients, for example).

Unfortunately, the `partial_fit` API doesn't play that nicely with my favorite
part of scikit-learn:
[pipelines](http://scikit-learn.org/stable/modules/pipeline.html#pipeline),
which we discussed at length in [part 1](scalable-ml-01). You would essentially
need every step in the pipeline to have an out-of-core `parital_fit` version,
which isn't really feasible; some algorithms just have to see the entire dataset
at once. Setting that aside, it wouldn't be great for a user, since working
with generators of datasets is awkward compared to the expressivity we get from
pandas and NumPy.

Fortunately, we *have* great data containers for larger than memory arrays and
dataframes: `dask.array` and `dask.dataframe`. We can

1. Use dask for pre-processing data in an out-of-core manner
2. Use scikit-learn to fit the actual model, out-of-core, using the
   `partial_fit` API

And all of this can be done in a pipeline. The rest of this post shows how.

## Big Arrays

If you follow along in [companion notebook][notebook], you'll see that I
generate a dataset, replicate it 100 times, and write the results out to
parquet. I then read it back in as a pair of `dask.dataframe`s and convert them
to a pair of `dask.array`s. I'll skip those details to focus on main goal: using
`sklearn.Pipeline`s on larger-than-memory datasets. Suffice to say, we have a
function `read` that gives us our big `X` and `y`:

```python
X, y = read()
X
```

    dask.array<concatenate, shape=(100000000, 20), dtype=float64, chunksize=(500000, 20)>
  
  
```python
y
```

    dask.array<squeeze, shape=(100000000,), dtype=float64, chunksize=(500000,)>


So X is a 100,000,000 x 20 array of floats that we'll use to predict `y`. I
generated the dataset, so I know that `y` is either 0 or 1. We'll be doing
classification.

```python

(X.nbytes + y.nbytes) / 10**9
```

    16.8


My laptop has 16 GB of RAM, and the dataset is 16.8 GB. We can't simply read the
entire thing into memory. We'll use dask for the preprocessing, and scikit-learn
for the fitting. To demonstrate the idea, we'll have a small pipeline

1. Scale the features by mean and variance
2. Fit an SGDClassifer

I've implemented a `daskml.preprocessing.StandardScaler``, using dask, in about
40 lines of code. This will operate completely in parallel and out-of-core.

I *haven't* implemented a custom `SGDClassifier`, because that'd be much more than
40 lines of code. I have a small wrapper that will use scikit-learn's
implementation to provide fit method that operates out-of-core, but not in
parallel.

```
from daskml.preprocessing import StandardScaler
from daskml.linear_model import BigSGDClassifier

from dask.diagnostics import ResourceProfiler, Profiler, ProgressBar
from sklearn.pipeline import make_pipeline
```

As a user, the API is the same as `scikit-learn`. Indeed, it *is* just a regular
`sklearn.pipeline.Pipeline`.

```python
pipe = make_pipeline(
    StandardScaler(),
    BigSGDClassifier(classes=[0, 1], max_iter=1000, tol=1e-3, random_state=2),
)
```

And fitting is identical as well: `pipe.fit(X, y)`. We'll collect some
performance metrics as well.

```python
%%time
rp = ResourceProfiler()
p = Profiler()


with p, rp:
    pipe.fit(X, y)
```

    CPU times: user 2min 38s, sys: 1min 44s, total: 4min 22s
    Wall time: 1min 47s

At this point, `pipe` has all the regular methods you would expect, ``predict``,
``predict_proba``, etc. You can get to the individual attributes like
``pipe.steps[1][1].coef_``.

One important point to stress here: when we get to the `BigSGDClassifier.fit`
at the end of the pipeline, everything is done serially. We can see that by
plotting the `Profiler` we captured up above:

![](images/sml-02-fit.png)

That graph shows the tasks each worker (a core on my laptop) executed over time.
Towards the start, when we're reading off disk, converting to `dask.array`s, and
doing the `StandardScaler`, everything is in parallel. Once we get to the
`BigSGDClassifier`, which is just a simple wrapper around
`sklearn.linear_model.SGDClassifier`, we lose all our parallelism*.

The predict step *is* done entirely in parallel.

```python

with rp, p:
    predictions = pipe.predict(X)
    predictions.to_dask_dataframe(columns='a').to_parquet('predictions.parq')

```

![](images/sml-02-predict.png)

When I had this idea last week, of feeding blocks of `dask.array` to a
scikit-learn estimator's `partial_fit` method, I thought it was pretty neat.
Turns out Matt Rocklin had the idea, and implemented it in dask, two years ago.

Roughly speaking, the implementation is:


```python
class BigSGDClassifier(SGDClassifer):
    ...
    
    def fit(self, X, y):
        # ... some setup
        for xx, yy in by_blocks(X, y):
            self.parital_fit(xx, yy)
        return self
```

We iterate over the dask arrays block-wise, and pass it into the estimators
`parital_fit` method.

Let me know what you think. I'm pretty excited about this because it
removes some of the friction around using sckit-learn Pipelines with
out-of-core estimators. I'll be packaging this up in `daskml` to make it more
usable for the community over the next couple weeks.

[notebook]: http://nbviewer.jupyter.org/github/TomAugspurger/scalable-ml/blob/master/partial.ipynb
