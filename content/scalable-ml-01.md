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
constraints. How are we constrained by scale? The two main ones I can think of
are

1. I'm constrained by size: I can't fit my model on my entire dataset using my
   laptop. I'd like to scale out by adopting algorithms that work in batches
   locally, or on a distributed cluster.
2. I'm constrained by time: I'd like to fit more models on my dataset in a given
   amount of time. I'd like to scale out by fitting more models in parallel,
   either on my laptop by using more cores, or on a cluster.

These aren't mutually exclusive or exhaustive, but they should serve as a nice
guide for our discussion. I'll be showing where the usual pandas + scikit-learn
for in-memory analytics workflow breaks down, and offering solutions (some of
which don't exist as I write this).

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

and fit your model on a (representative) subset of your data. *You may not need
distributed machine learning*. The tricky thing is selecting how large your
sample should be. The "correct" value depends on the complexity of your learning
task, the complexity of your model, and the nature of your data. The best you
can do here is think carefully about your problem, and to plot the [learning
curve].

![scikit-learn](http://scikit-learn.org/stable/_images/sphx_glr_plot_learning_curve_001.png)

<div style="text-align: center"> <a
  href="http://scikit-learn.org/stable/_images/sphx_glr_plot_learning_curve_001.png"><i>source</i></a>
</div>

As usual, the scikit-learn developers do a great job explaining the concept in
addition to providing a great library. I encourage you to follow [that
link][learning curve]. This gist is that, for some models on some datasets
anyways, training on additional data past a point doesn't improve the model's
performance. At some point the learning curve levels off and you're just wasting
time and money training on those extra observations.

For today, we'll assume that we're on the flat part of the learning curve. Later
in the series we'll explore cases where we run out of RAM before the learning
curve levels off.

## Fit, Predict

In my experience, the first place I bump into RAM constraints is when I have a
manageable training dataset to fit the model on, but I have to make predictions
for a dataset that's orders of magnitude larger. In these cases, I fit my model
like normal, and predict in an out-of-core fashion. We'll see how dask allows us
to write normal-looking code, so we don't have to worry about writing the
batching code ourself.

To make this concrete, we'll use the (tried and true) New York taxi cabs
dataset. The goal will be to predict if the passenger tips (but that's *really*
not the point. The point is to explore a space here). We'll train the data on a
single month's worth of data (which fits in my laptop's RAM), and predict on the
full dataset[^2].

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

<table> <thead> <tr style="text-align: right;"> <th></th> <th>vendor_name</th>
  <th>Trip_Pickup_DateTime</th> <th>Trip_Dropoff_DateTime</th>
  <th>Passenger_Count</th> <th>Trip_Distance</th> <th>Start_Lon</th>
  <th>Start_Lat</th> <th>Rate_Code</th> <th>store_and_forward</th>
  <th>End_Lon</th> <th>End_Lat</th> <th>Payment_Type</th> <th>Fare_Amt</th>
  <th>surcharge</th> <th>mta_tax</th> <th>Tip_Amt</th> <th>Tolls_Amt</th>
  <th>Total_Amt</th> </tr> </thead> <tbody> <tr> <th>0</th> <td>VTS</td>
  <td>2009-01-04 02:52:00</td> <td>2009-01-04 03:02:00</td> <td>1</td>
  <td>2.63</td> <td>-73.991957</td> <td>40.721567</td> <td>NaN</td> <td>NaN</td>
  <td>-73.993803</td> <td>40.695922</td> <td>CASH</td> <td>8.9</td> <td>0.5</td>
  <td>NaN</td> <td>0.00</td> <td>0.0</td> <td>9.40</td> </tr> <tr> <th>1</th>
  <td>VTS</td> <td>2009-01-04 03:31:00</td> <td>2009-01-04 03:38:00</td>
  <td>3</td> <td>4.55</td> <td>-73.982102</td> <td>40.736290</td> <td>NaN</td>
  <td>NaN</td> <td>-73.955850</td> <td>40.768030</td> <td>Credit</td>
  <td>12.1</td> <td>0.5</td> <td>NaN</td> <td>2.00</td> <td>0.0</td>
  <td>14.60</td> </tr> <tr> <th>2</th> <td>VTS</td> <td>2009-01-03 15:43:00</td>
  <td>2009-01-03 15:57:00</td> <td>5</td> <td>10.35</td> <td>-74.002587</td>
  <td>40.739748</td> <td>NaN</td> <td>NaN</td> <td>-73.869983</td>
  <td>40.770225</td> <td>Credit</td> <td>23.7</td> <td>0.0</td> <td>NaN</td>
  <td>4.74</td> <td>0.0</td> <td>28.44</td> </tr> <tr> <th>3</th> <td>DDS</td>
  <td>2009-01-01 20:52:58</td> <td>2009-01-01 21:14:00</td> <td>1</td>
  <td>5.00</td> <td>-73.974267</td> <td>40.790955</td> <td>NaN</td> <td>NaN</td>
  <td>-73.996558</td> <td>40.731849</td> <td>CREDIT</td> <td>14.9</td>
  <td>0.5</td> <td>NaN</td> <td>3.05</td> <td>0.0</td> <td>18.45</td> </tr> <tr>
  <th>4</th> <td>DDS</td> <td>2009-01-24 16:18:23</td> <td>2009-01-24
  16:24:56</td> <td>1</td> <td>0.40</td> <td>-74.001580</td> <td>40.719382</td>
  <td>NaN</td> <td>NaN</td> <td>-74.008378</td> <td>40.720350</td> <td>CASH</td>
  <td>3.7</td> <td>0.0</td> <td>NaN</td> <td>0.00</td> <td>0.0</td>
  <td>3.70</td> </tr> </tbody> </table>


This takes about a minute on my laptop. The dataset has about 14M rows. Like I
said, the training step is going to be completely normal. We'll do the usual
train-test split, followed by some totally normal data pre-processing:

```python
X = df.drop("Tip_Amt", axis=1)
y = df['Tip_Amt'] > 0

X_train, X_test, y_train, y_test = train_test_split(X, y)
```

This isn't a perfectly clean dataset, which is nice because it gives us a chance
to demonstrate some of pandas' pre-processing prowess, before we hand the data
of to scikit-learn to fit the model. Since we're operating with scale in mind,
we'll be extremely cautious to perform *all* the data transformations inside a
`Pipeline`. This has many benefits, but the main one for our purpose today is
that it packages our entire task into a single python object. Later on, our
`predict` step will be a single function call.


```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin
```

If you're unfamiliar with pipelines, check out the [scikit-learn
docs][pipelines-docs], [this blog][pipelines-blog] post, and my talk from
[PyData Chicago 2016][pipelines-pandas]. The short version: a pipeline consists
of multiple steps. Each step transforms the data somehow, and the final step is
a normal `Estimator` like `LogisticRegression`.

I notice that there are some minor differences in the spelling on "Payment
Type":

```python
df.Payment_Type.cat.categories
```

    Index(['CASH', 'CREDIT', 'Cash', 'Credit', 'Dispute', 'No Charge'], dtype='object')

We'll reconcile that by lower-casing everything with a `.str.lower()`. But
resist the temptation to just do that imperatively inplace! We'll package it up
into a function that will later be wrapped up in a [FunctionTransformer].

```python
def payment_lowerer(X):
    return X.assign(Payment_Type=X.Payment_Type.str.lower())
```

I should note here that I'm using
[`.assign`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.assign.html)
to update the variables since it's implicitly copies the data. We don't want to
be modifying the caller's data without their consent.

Not all the columns look useful. We could have easily solved this by only
reading in the data that we're actually going to use, but let's solve it now
with another simple transformer:

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

Under the hood, pandas stores `datetimes` like `Trip_Pickup_DateTime` as a
64-bit integer representing the nanoseconds since some time in the 1600s. If we
left this untransformed, scikit-learn would happily transform it to its integer
representation, which *may* not be the most meaningful item to stick in a linear
model for predicting tips. A better feature might the hour of the day:

```python
class HourExtractor(TransformerMixin):
    "Transform each datetime64 column in `columns` to integer hours"
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.assign(**{col: lambda x: x[col].dt.hour
                           for col in self.columns})
```

Likewise, we'll need to ensure the categorical variables (in a statistical
sense) are categorical dtype (in a pandas sense).

```python
class CategoricalEncoder(TransformerMixin):
    """
    Convert to Categorical with specific `categories`.
    
    Examples
    --------
    >>> CategoricalEncoder({"A": ['a', 'b', 'c']}).fit_transform(
    ...     pd.DataFrame({"A": ['a', 'b', 'a', 'a']})
    ... )['A']
    0    a
    1    b
    2    a
    3    a
    Name: A, dtype: category
    Categories (2, object): [a, b, c]
    """
    def __init__(self, categories):
        self.categories = categories
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X = X.copy()
        for col, categories in self.categories.items():
            X[col] = X[col].astype('category').cat.set_categories(categories)
        return X
```

Incidentally, my [`CategoricalDtype` pull
request](https://github.com/pandas-dev/pandas/pull/16015) will make this type of
operation a bit easier to express.

Finally, we'd like to scale a subset of the data. Scikit-learn has a
`StandardScaler`, which we'll mimic here, to just operate on a subset of the
columns.

```python
class StandardScaler(TransformerMixin):
    "Scale a subset of the columns in a DataFrame"
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        # Yes, non-ASCII symbols can be a valid identfiers in python 3
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
data. It'll make our lives easier in the `.predict` step later on.

We can fit the pipeline as normal:

```python
%time pipe.fit(X_train, y_train)
```

This take about a minute on my laptop. We can check the accuracy (but again,
this isn't the point)

```python
>>> pipe.score(X_train, y_train)
0.9931

>>> pipe.score(X_test, y_test)
0.9931
```

It turns out people essentially tip if and only if they're paying with a card,
so this isn't a particularly difficult task. Or perhaps more accurately, tips
are only *recorded* when someone pays with a card.


## Scaling Out

OK, so we've fit our model and it's been basically normal. Maybe we've been a
overly-dogmatic about doing *everything* in a pipeline, but that's just good
model hygiene anyway.

Now, to scale out to the rest of the dataset. We'll predict the probability of
tipping for every cab ride in the dataset (bearing in mind that the full dataset
doesn't fit in my laptop's RAM).

To make things a bit easier we'll use dask, though it isn't strictly necessary
for this section. It saves us from writing a for loop (big whoop). Additionally,
we'll be able to reuse this code when we go to scale out to a cluster (that part
is pretty cool, actually). Dask can scale down to a single laptop, and up to
thousands of cores.

```python
import dask.dataframe as dd

df = dd.read_csv("data/*.csv", dtype=dtype,
                 parse_dates=['Trip_Pickup_DateTime', 'Trip_Dropoff_DateTime'],)

X = df.drop("Tip_Amt", axis=1)
```

`X` is a `dask.dataframe`, which can be mostly be treated as a single dataframe,
and thought of as a collection of many smaller pandas `DataFrames`.

Since scikit-learn isn't dask-aware, we can't simply call
`pipe.predict_proba(X)`. At some point, our `dask.dataframe` would be cast to a
`numpy.ndarray`, and our memory would blow up. Fortunately, dask has a some nice
little escape hatches for dealing with functions that know how to operate on
NumPy arrays, but not dask objects. In this case, we'll use `map_partitions`.

```python
yhat = X.map_partitions(lambda x: pd.Series(pipe.predict_proba(x)[:, 1],
                                            name='yhat'),
                        meta=('yhat', 'f8'))
```

`map_partitions` will go through each partition in your dataframe (one per
file), calling the function on each partition. Dask worries about stitching
together the result (though we provide a hint with the `meta` keyword, to say
that it's a `Series` with name `yhat` and dtype `f8`).

Now we can write it out to disk (using parquet rather than CSV, because CSVs are
evil)

```python
yhat.to_frame().to_parquet("data/predictions.parq")
```

This takes about 9 minutes to finish on my laptop.

## Scaling Out (even further)

If 9 minutes is too long, and you happen to have a cluster sitting around, you
can repurpose that dask code to run on the [distributed scheduler]. I'll use
`dask-kubernetes`, to start up a cluster but you may already have access to one
from your business or institution.

```python
dask-kubernetes create scalable-ml
```

This sets up a cluster with 8 workers and 54 GB of memory.

The next part of this post is a bit fuzzy, since your teams will probably have
different procedures and infrastructure around persisting models. At my old job,
I wrote a small utility for serializing a scikit-learn model along with some
metadata about what it was trained on, before dumping it in S3. If you want to
be fancy, you should watch [this talk](https://www.youtube.com/watch?v=vKU8MWORHP8)
by [Rob Story](twitter.com/oceankidbilly) on how Stripe handles these things
(it's a bit more sophisticated that my "dump it on S3" script).

For this blog post, "shipping it to prod" consists of a `joblib.dump(pipe,
"taxi-model.pkl")` on our laptop, and copying it to somewhere the cluster can
load the file. Then on the cluster, we'll load it up, and create a `Client` to
communicate with our cluster's workers.

```python
from distributed import Client
from sklearn.externals import joblib

pipe = joblib.load("taxi-model.pkl")
c = Client('dask-scheduler:8786')
```

Depending on how your cluster is set up, specifically with respect to having a
shared-file-system or not, the rest of the code is more-or-less identical. If
we're using S3 or gcfs as our shared file system, we'd modify the loading code
to read from S3, rather than our local hard drive:

```python
df = dd.read_csv("s3://bucket/yellow_tripdata_2009*.csv",
                 dtype=dtype,
                 parse_dates=['Trip_Pickup_DateTime', 'Trip_Dropoff_DateTime'],
                 storage_options={'anon': True})
df = c.persist(df)  # persist the dataset in distributed memory
                    # across all the workers in the Dataset
X = df.drop("Tip_Amt", axis=1)
y = df['Tip_Amt'] > 0
```

Computing the predictions is identical to our out-of-core-on-my-laptop code:


```python
yhat = X.map_partitions(lambda x: pd.Series(pipe.predict_proba(x)[:, 1], name='yhat'),
                        meta=('yhat', 'f8'))
```

And saving the data (say to S3) might look like

```python
yhat.to_parquet("s3://bucket/predictions.parq")
```

The loading took about 4 minutes on the cluster, the predict about 10 seconds,
and the writing about 1 minute. Not bad overall.

## Wrapup

Today, we went into detail on what's potentially the first scaling problem
you'll hit with scikit-learn: you can train your dataset in-memory (on a laptop,
or a large workstation), but you have to predict on a much larger dataset.

We saw that the existing tools handle this case quite well. For training, we
followed best-practices and did everything inside a `Pipeline` object. For
predicting, we used `dask` to write regular pandas code that worked out-of-core
on my laptop or on a distributed cluster.

If this topic interests you, you should watch [this talk](scaling sklearn) by
[Stephen Hoover] on how Civis is scaling scikit-learn.

In future posts we'll dig into

- how dask can speed up your existing pipelines by executing them in parallel
- scikit-learn's out of core API for when your training dataset doesn't fit in
  memory
- using dask to implement distributed machine learning algorithms

Until then I would *really* appreciate your feedback. My personal experience
using scikit-learn and pandas can't cover the diversity of use-cases they're
being thrown into. You can reach me on Twitter
[@TomAugspurger](https://twitter.com/TomAugspurger) or by email at
<mailto:tom.w.augspurger@gmail.com>. Thanks for reading!

[dask]: https://dask.pydata.org
[learning curve]: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
[tpot]: http://rhiever.github.io/tpot/
[auto-sklearn]: http://automl.github.io/auto-sklearn/
[pipelines-docs]: http://scikit-learn.org/stable/modules/pipeline.html#pipeline
[pipelines-blog]: http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
[pipelines-pandas]: https://www.youtube.com/watch?v=KLPtEBokqQ0
[FunctionTransformer]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html
[scaling sklearn]: https://www.youtube.com/watch?v=KqKEttfQ_hE
[Stephen Hoover]: https://twitter.com/stephenactual
[distributed scheduler]: http://distributed.readthedocs.io/en/latest/

[^*]: p < .05
[^2]: This is a bad example, since there could be a time-trend or seasonality to
    the dataset. But our focus isn't on building a good model, I hope you'll
    forgive me.
