---
title: "Rewriting scikit-learn for big data, in under 9 hours."
date: 2018-01-28
slug: dask-ml-iid
---

This past week, I had a chance to visit some of the scikit-learn developers at
Inria. It was a fun and productive week, and I'm thankful to them for hosting me
and Anaconda for sending me there.

Towards the end of our week, Gael threw out the observation that for many
applications, you don't need to *train* on the entire dataset. But it'd be nice
if the trained estimator would be able to *transform* and *predict* for dask
arrays, getting all the nice distributed parallelism and memory management dask
brings.

This intrigued me, and I had a 9 hour plane ride, so...

## ``dask_ml.iid``

I put together the ``dask_ml.iid`` sub-package. The estimators contained within
are appropriate for data that are independently and identically distributed
(IID). Roughly speaking, your data is IID if there aren't any "patterns" in the
data as you move top to bottom. For example, time-series data is often *not*
IID, there's often an underlying time trend to the data. Or the data may be
autocorrelated (if `y` was above average yesterday, it'll probably be above
average today too). If your data is sorted by, say, customer ID, then it likely
isn't IID. You might be able to shuffle it in this case.

If your data are IID, it *may* be OK to just fit the model on the first block.
In principal, it should be a representative sample of your entire dataset.

Here's a quick example. We'll fit a `GradientBoostingClassifier`. The dataset
will be 1,000,000 x 20, in chunks of 10,000. This would take *way* too long to
fit regularly. But, with IID data, we may be OK fitting the model on just the
the first 10,000 observations.

```python
>>> from dask_ml.datasets import make_classification
>>> from dask_ml.iid.ensemble import GradientBoostingClassifier

>>> X, y = make_classification(n_samples=1_000_000, chunks=10_100)

>>> clf = GradientBoostingClassifier()
>>> clf.fit(X, y)
```

At this point, we have a scikit-learn estimator that can be used to transform or
predict for dask arrays, in parallel.

```python
>>> prob_a
dask.array<predict_proba, shape=(1000000, 2), dtype=float64, chunksize=(10000, 2)>

>>> prob_a[:10].compute()
array([[0.98268198, 0.01731802],
       [0.41509521, 0.58490479],
       [0.97702961, 0.02297039],
       [0.91652623, 0.08347377],
       [0.96530773, 0.03469227],
       [0.94015097, 0.05984903],
       [0.98167384, 0.01832616],
       [0.97621963, 0.02378037],
       [0.95951444, 0.04048556],
       [0.98654415, 0.01345585]])
```

An alternative to this is to sample your data and use a regular scikit-learn
estimator. But the `dask_ml.iid` approach is *slightly* preferable, since
post-fit tasks like prediction can be done on dask arrays in parallel (and
potentially distributed). Scikit-Learn's estimators are not dask-aware, so
they'd just convert it to a NumPy array, possibly blowing up your memory.

If dask and `dask_ml.iid` had existed a few years ago, it would have solved all
the "big data" needs of my old job. Personally, I never hit a problem where, if
my dataset was already large, training on an even larger dataset was the answer.
I'd always hit the level part of the learning curve, or was already dealing with
highly imbalanced classes. But, I would often have to make *predictions* for a
much larger dataset. For example, I might have trained a model on "all the
customers for this store" and predicted for "All the people in Iowa".

## The Implementation

Just because I think it's cool, here's a bit about how it's done.

`dask_ml.iid` doesn't *really* have a `class` definition inside it. Instead, we
dynamically create the classes using the `type` builtin. You may be familiar
with using `type` to get the type of an object:

```python
>>> type(2)
int
```

But, `type` has a second signature

```python
type(name : str, bases : Tuple[Type], Dict) -> Type
```

This let's you create new classes on the fly. The only trick is getting the
correct `bases`, the list of base classes for our new type. We implemented a few
mixin classes like `IIDMixin` that define `fit` to

1. Grab just the first block of `X` and `y`
2. Call `super().fit` on the first blocks
3. Copy over the learned attributes

Methods like `predict` are similarly mixed and (roughly) just do
`X.map_partitions(super().predict)`.
