---
title: Easy distributed training with Joblib and `dask.distributed`
date: 2017-10-26
slug: distributed-joblib
status: draft
---

Scikit-learn uses [joblib]() for simple parallelism in many places, anywhere you
see an ``n_jobs`` keyword. Your estimator may have an embarrassingly parallel
step internally (fitting each of the trees in a `RandomForest` for example). Or
your meta-estimator like `GridSearchCV` may try out many combinations of
hyper-parameters in parallel.

Joblib offers a few "backends" for how to do your parallelism, but they all boil
down to "does it use many threads, or many processes?"

## Parallelism in Python

A quick digression on *single-machine* parallelism in Python. We have two main
choices: multiple threads or multiple processes. We can't say up front that one
is always better than the other; it unfortunately depends on the specific
workload. But we do have some general heuristics, that come down to
serialization overhead and Python's Global Interpreter Lock (GIL).

The GIL is part of CPython's (i.e. the C program that interprets and runs your
Python program). It limits your process so that only one thread is running
*Python* code at once. Fortunately, much of the numerical Python stack is
written in C, Cython, or C++, and *may* be able to "release" the GIL. This means
your "Python" program, which is calling into Cython or C via NumPy or pandas,
can get real thread-based parallelism without being limited by the GIL. The main
caveat here is working with string data or Python objects (lists, dicts, sets,
etc). Since those are touching Python objects, NumPy, pandas, etc. will still
need to hold the GIL to manipulate them.

So, if we have the *option* of using threads instead of processes, which do we
choose? For most numeric / scientific workloads, threads are better than
processes because of *shared memory*. Each thread in a thread-pool can view (and
modify!) the *same* large NumPy array or pandas dataframe. With multiple
processes, data must be *serialized* between processes (perhaps using pickle).
For large arrays or dataframes this can be slow, and it may blow up your memory
if the data a decent fraction of your machine's RAM. You'll have a full copy in
each processes.

## Changes to Joblib

A while back, Jim Crist added the ``dask.distributed.joblib`` backend for
joblib. ``dask.distributed`` registers a backend with joblib, so that users can
parallelize a computation across an *entire cluster*, not just your local
machine's threads or processes.

This is great when

1. Your dataset is not too large (since the data must be sent to each worker)
2. The runtime of each task (say fitting one of the trees in a `RandomForest`
   is long enough that the overhead of serializing the data across the network
   to the worker doesn't dominate the runtime
3. You have *many* parallel tasks to run (else, you'd just use a local thread or
   process pool and avoid the network delay)

Here's a small example showing how to use dask's distributed joblib backend.

We will

1. Create / connect to our `dask.distributed` cluster (normally you'd put the IP
   address of your scheduler in the call to `Client`)
2. Define a function to simulate our "work", which draws a random number, and
   sleeps for that times a factor
3. Do the work in parallel, using a `joblib.Parallel` call inside our parallel
   backend call.

```python
import random
import time

import joblib
import distributed.joblib  # register backend
from dask.distributed import Client


client = Client()  # Pass the IP address of your scheduler here.


def f(x):
    result = x * random.random()
    time.sleep(result)  # simulate a long computation
    return result


with joblib.parallel_backend("dask.distributed"):
    # 'results' is computed in parallel using every worker
    # in your cluster.
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(f)(i) for i in range(10)
    )
```

Each call to `f` happens on one of our workers. With a large enough cluster,
each call would happen on a different worker.

There were a few issues with this though, which we were able to resolve last
week.

First, `dask.distributed`'s joblib backend didn't handle *nested* parallelism
well. This may occur if you do something like

```pytohn
gs = GridSearchCV(Estimator(n_jobs=-1), n_jobs=-1)
```

You could run into deadlocks where the outer level kicks off a bunch of
`Parallel` calls. Those inner `Parallel` calls would make their way to the
distributed scheduler, who would look around for a free worker. But all the
workers were "busy" waiting around for the outer `Parallel` calls to finish,
which weren't progressing because there weren't any free workers! Deadlock!

`dask.distributed` has a solution for this case (workers `secede` from the
thread pool when they start a long-running `Parllel` call, and `rejoin` when
they're done), but we needed a way to negotiate with joblib about when the
`secede` and `rejoin` should happen. Joblib now has an API for backends to
control some setup and teardown around the actual function execution.

Second, some places in scikit-learn hard-code the backend they want to use in
their `Parallel()` call, meaning the cluster isn't used. This may be because the
algorithm author knows that one backend performs better than others. For
example, `RandomForest.fit` performs better with threads, since it's purely
numeric and releases the GIL. In this case we would say the `Parallel` call
*prefers* threads, since you'd get the same result with processes, it'd just be
slower.

Another reason for hard-coding the backend is if the *correctness* of the
implementation relies on it. For example, `RandomForest.predict` allocates the
output array and mutates the output array from many threads (it knows not to
mutate the same place). In this case, we'd say the `Parallel` call *requires*
shared memory, because you'd get an incorrect result using processes.

The solution was enhance `joblib.Parallel` to take two new keywords, `prefer`
and `require`. If a `Parallel` call *prefers* threads, it'll use them, unless
it's in a context saying "use this backend instead", like

```python
with joblib.parallel_backend('dask.distributed'):
    # This uses dask's workers, not threads
    joblib.Parallel(n_jobs=-1, prefer="threads")(...)
```

On the other hand, if a `Parallel` requires a specific backend, it'll get it.

```python
with joblib.parallel_backend('dask.distributed'):
    # This uses the threading backend, since shared memory is required
    joblib.Parallel(n_jobs=-1, require="sharedmem")(...)
```

This is a elegant way to negotiate a compromise between

1. The *user*, who knows best about what resources are available, as specified
   by the `joblib.parallel_backend` context manager. And,
2. The *algorithm author*, who knows best about the GIL handling and shared
   memory requirements.

After the next joblib release, we'll update scikit-learn to use it in places
where it currently hardcodes the parallel backend.

#TODO: embed movie of the distributed scheduler.
