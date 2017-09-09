---
title: "Scalable Machine Learning (Part 1)"
date: 2016-09-06
slug: scalable-ml-01
status: draft
---

The [dask] project is interested in scaling the scientific python ecosystem to
larger datasets. My current focus is on out-of-core, parallel, and distributed
machine learning. This series will introduce those concepts, explore what we
have available today, and track the community's efforts to push the boundaries.

## Constraints

I am (or was, anyway) an economist, and economists like to think in terms of
constraints. In what ways are we constrained by scale? The two main ones I can
think of are

1. I'm constrained by time: I'd like to fit more models on my dataset in a given
   amount of time. I'd like to scale out by fitting more models in parallel,
   either on my laptop by using more cores, or on a cluster.
2. I'm constrained by size: I can't fit my model on my entire dataset using my
   laptop. I'd like to scale out by adopting algorithms that work in batches
   locally, or on a distributed cluster.

These aren't mutually exclusive or exhaustive, but they should serve as a nice
starting point for our discussion.

## Scaling, with Dask

To quote the dask docs:

> Dask is a flexible parallel computing library for analytic computing.

It may not be immediately obvious why a library for parallel computing should
also be great for out-of-core *and* distributed computing.

## Don't forget your Statistics

Statistics is a thing[^*]. Statisticians have thought a lot about things like
sampling, and the variance of estimators. So it's worth stating up front that
you may be able to just

```sql
SELECT *
FROM dataset
ORDER BY random()
LIMIT 10000;
```

and fit your model on a (representative) subset of your data. The tricky thing
is selecting how large your sample should be. The "correct" value depends on the
complexity of your learning task, the complexity of your model, and the nature
of your data. The best you can do here is think carefully about your problem,
and to plot the [learning curve].

![scikit-learn](http://scikit-learn.org/stable/_images/sphx_glr_plot_learning_curve_001.png)

*http://scikit-learn.org/stable/_images/sphx_glr_plot_learning_curve_001.png*

As usual, the scikit-learn developers do a great job explaining the concept, in
addition to providing a great library. I encourage you to follow [that
link](learning curve). But the gist is that, for some models anyways, having
more data doesn't really improve the model's performance. At some point the
learning curve levels off, and you're just wasting energy with those extra
observations.

Throughout the rest of this series, we'll assume that we're on the
still-increasing part of the learning curve.

## Fit, Predict

In my experience, the first place I bump into RAM constraints is when I have a
manageable training dataset to fit the model on, but I have to make predictions
for a dataset that's orders of magnitude larger.

To make this concrete, we'll use the (tired, but well-known) New York taxi cabs
dataset. The goal will be to predict if the passenger tips (but that's *really*
not the point). We'll train the data on a single month's worth of data, and
predict on the full dataset[^2].

First, let's load in the first month of data from disk:

```python
dtype = {
    'vendor_name': 'category',
    'Payment_Type': 'category',
}

df = pd.read_csv("data/yellow_tripdata_2009-01.csv", dtype=dtype,
                 parse_dates=['Trip_Pickup_DateTime', 'Trip_Dropoff_DateTime'],)
df.head()
```

This takes about a minute on my laptop. The dataset has about 14M rows.

```python
X = df.drop("Tip_Amt", axis=1)
y = df['Tip_Amt'] > 0

X_train, X_test, y_train, y_test = train_test_split(X, y)
```

This isn't a perfectly clean dataset, which is nice because it gives us a chance
to demonstrate some of pandas' pre-processing prowess, before we hand the data
of to scikit-learn to fit the model. Since we're operating with scale in mind,
we'll be extremely cautions to perform *all* the data transformations inside a
`Pipeline`. This has many benefits, but the main one for our purpose today is
that it packages our entire task into a single python object.


```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin
```

If you're unfamiliar with pipelines, check out the [scikit-learn
docs][pipelines-docs], [this blog][pipelines-blog] post, and my talk from
[PyData Chicago 2016][pipelines-pandas]. The short version is that a pipeline
consists of multiple transformers, and ends in a normal `Estimator` like
`LogisticRegression`.

I notice that there are some minor differences in the spelling on "Payment Type":

```python
df.Payment_Type.cat.categories
```

    Index(['CASH', 'CREDIT', 'Cash', 'Credit', 'Dispute', 'No Charge'], dtype='object')

We'll reconcile that by lower-casing everything with a `.str.lower()`. But resist
the temptation to just do that imperatively inplace! We'll package it up into a
transform:

```python
def payment_lowerer(X):
    return X.assign(Payment_Type=X.Payment_Type.str.lower())
```

Later on we'll wrap this in a [`FunctionTransformer`][FunctionTransformer]

Not all the columns look useful. We could have easily solved this by only
reading in the data that we're actually going to use, but let's solve it now
with another transformer:

```python
class ColumnSelector(TransformerMixin):
    "Select `columns` from `X`"
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]
```

We can't stick datetimes in a model, so we'll extract the hour of the day and
use that as a feature.

```python
class HourExtractor(TransformerMixin):
    "Transform each datetime64 column in `columns` to integer hours"
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.assign(**{col: lambda x: x[col].dt.hour for col in self.columns})
```

Likewise, we'll need to ensure the categorical (in a statistical sense) are
categorical dtype (in a pandas sense).

```python
class CategoricalEncoder(TransformerMixin):
    "Convert to Categorical with specific `categories`."

    def __init__(self, categories):
        self.categories = categories
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        for col, categories in self.categories.items():
            X[col] = X[col].astype('category').cat.set_categories(categories)
        return X
```

Finally, we'd like to scale a subset of the data. Scikit-learn has a
`StandardScaler`, which we'll mimic here, to just operate on a subset of the
columns.

```python
class StandardScaler(TransformerMixin):
    "Scale a subset of the columns in a DataFrame"
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        self.μs = X[self.columns].mean()
        self.σs = X[self.columns].std()
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.columns] = X[self.columns].sub(self.μs).div(self.σs)
        return X
```

Now we can build up the pipeline:

```python
# The columns at the start of the pipeline
columns = ['vendor_name', 'Trip_Pickup_DateTime',
           'Passenger_Count', 'Trip_Distance',
           'Payment_Type', 'Fare_Amt', 'surcharge']

# The mapping of {column: set of categories}
categories = {
    'vendor_name': ['CMT', 'DDS', 'VTS'],
    'Payment_Type': ['cash', 'credit', 'dispute', 'no charge'],
}

scale = ['Trip_Distance', 'Fare_Amt', 'surcharge']

pipe = make_pipeline(
    ColumnSelector(columns),
    HourExtractor(['Trip_Pickup_DateTime']),
    FunctionTransformer(payment_lowerer, validate=False),
    CategoricalEncoder(categories),
    FunctionTransformer(pd.get_dummies, validate=False),
    StandardScaler(scale),
    LogisticRegression(),
)
pipe
```

    [('columnselector', <__main__.ColumnSelector at 0x1a2c726d8>),
     ('hourextractor', <__main__.HourExtractor at 0x10dc72a90>),
     ('functiontransformer-1', FunctionTransformer(accept_sparse=False,
               func=<function payment_lowerer at 0x17e0d5510>, inv_kw_args=None,
               inverse_func=None, kw_args=None, pass_y='deprecated',
               validate=False)),
     ('categoricalencoder', <__main__.CategoricalEncoder at 0x11dd72f98>),
     ('functiontransformer-2', FunctionTransformer(accept_sparse=False,
               func=<function get_dummies at 0x10f43b0d0>, inv_kw_args=None,
               inverse_func=None, kw_args=None, pass_y='deprecated',
               validate=False)),
     ('standardscaler', <__main__.StandardScaler at 0x162580a90>),
     ('logisticregression',
      LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
               intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
               penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
               verbose=0, warm_start=False))]

So, all that pipeline scaffolding is a *bit* of extra complexity over
imperatively updating the data or writing simple functions to transform the
data. It'll be worth it though, when we go to predict for the entire dataset.

We can fit the pipeline as normal:

```python
%time pipe.fit(X_train, y_train)
```

This take about a minute on my laptop. We can check the performance, but that's
not the point.


```python
>>> pipe.score(X_train, y_train)
0.9931

>>> pipe.score(X_test, y_test)
0.9931
```


It turns out people essentially tip if and only they're paying with a card, so
this isn't a particularly difficult task. Or perhaps more accurately, tips are
only *recorded* when someone pays with a card.

Now, to scale out to the rest of the dataset. We'll predict the probability of
tipping for every cab ride in the dataset (bearing in mind that the full dataset
doesn't fit in my laptop's RAM).

To make things a bit easier we'll use dask, though it isn't strictly necessary
for this section. It saves us from writing a for loop, but more importantly
we'll be able to reuse this code when we go to scale out to a cluster. This does
demonstrate dask's ability to scale down to a laptop and operate out-of-core on
large datasets.

```python
import dask.dataframe as dd

df = dd.read_csv("data/*.csv", dtype=dtype,
                 parse_dates=['Trip_Pickup_DateTime', 'Trip_Dropoff_DateTime'],)

X = df.drop("Tip_Amt", axis=1)
```

Since scikit-learn isn't dask-aware, we can't simply call `pipe.predict_proba(X)`.
At some point, our `dask.dataframe` would be cast to a `numpy.ndarray`, and our
memory would blow up. Fortunately, `dask.dataframe` has a nice little escape
hatch for dealing with functions that know how to operate on NumPy arrays, but
not dask objects: `map_partitions`.

```python
yhat = X.map_partitions(lambda x: pd.Series(pipe.predict_proba(x)[:, 1],
                                            name='yhat'),
                        meta=('yhat', 'f8'))
yhat.to_frame().to_parquet("data/predictions.parq")
```

`map_partitions` will go through each partition in your dataframe (one per
file), calling the function on each partition. Dask worries about stitching
together the result (though we provide a hint with the `meta` keyword, to say
that it's a `Series` with name `yhat` and dtype `f8`).

This takes about 9 minutes to finish on my laptop.

You may have noticed this, but to make it explicit: this final step can run
out-of-core on my single laptop, or it can operate on a distributed cluster.
This will be a common pattern throughout this series. Let's explore that now by
setting up a small cluster. I'll use `dask-kubernetes`, but you may already have
access to one from your business or institution.

```python
dask-kubernetes create scalable-ml
```

This sets up a cluster with 8 workers and 54 GB of memory. Once that's up, I
could pickle up the model and load it along with a `Client`:

```python
from distributed import Client
from sklearn.externals import joblib

pipe = joblib.load("taxi-model.pkl")
c = Client('dask-scheduler:8786')
```

Depending on how your cluster is set up, specifically with respect to having a
shared-file-system or not, the rest of the code is more-or-less identical. If
we're using S3 or gcfs as our shared file system, we'd modify the code as


```python
df = dd.read_csv("s3://bucket/yellow_tripdata_2009*.csv",
                 dtype=dtype,
                 parse_dates=['Trip_Pickup_DateTime', 'Trip_Dropoff_DateTime'],
                 storage_options={'anon': True})
df = c.persist(df)
X = df.drop("Tip_Amt", axis=1)
y = df['Tip_Amt'] > 0
```

to load the data, and


```python
yhat = X.map_partitions(lambda x: pd.Series(pipe.predict_proba(x)[:, 1], name='yhat'),
                        meta=('yhat', 'f8'))
yhat.to_parquet("s3://bucket/predictions.parq")
```

To compute and store it. The loading took about 4 minutes on the cluster, the
predict about 10 seconds, and the writing about 1 minute. Not bad overall.

---

Scratch material to find a home for:

## Axes, for Scale

Dask can "scale out" in a couple dimensions. In 
I've found it use it useful to mentally bucket things into three groups:

1. out-of-core
2. parallel
3. distributed

First, parallelism. In the goal of "minimize some objective function", there are
many opportunities to parallelize computation. At the highest-level, we may be
using some kind grid search or ensemble method, which have embarrassing
parallelism baked into them. We attempt two values of a hyper-parameter at the
same time. A dask-powered machine learning library should interact well with
libraries like [auto-sklearn] and [tpot] that use this high-level parallelism.

Algorithms, too, can have parallelism.

Finally, the low-level optimizers can work in parallel too.

These various levels provides opportunities, but raise the specter *nested
parallelism*.

## Existing Landscape

Scikit-learn offers a a ``partial_fit`` API for out-of-core machine learning.
This should probably be your first stop when attempting to scale out a model.

[dask]: https://dask.pydata.org
[learning curve]: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
[tpot]: http://rhiever.github.io/tpot/
[auto-sklearn]: http://automl.github.io/auto-sklearn/
[pipelines-docs]: http://scikit-learn.org/stable/modules/pipeline.html#pipeline
[pipelines-blog]: http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
[pipelines-pandas]: https://www.youtube.com/watch?v=KLPtEBokqQ0
[FunctionTransformer]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html

[^*]: p < .05
[^2]: This is a bad example, since there could be a time-trend or seasonality to
    the dataset. But our focus isn't on building a good model, I hope you'll
    forgive me.
