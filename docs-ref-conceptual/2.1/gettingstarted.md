---
title: Getting started
ms.date: 05/31/2017
ms.topic: get-started-article
ms.service: cognitive-toolkit
---

# Getting started 

You can optionally try the [tutorials](https://notebooks.azure.com/cntk/libraries/tutorials) with pre-installed CNTK running in Azure Notebook hosted environment (for free) if you have not installed the toolkit in your own machine.

> If you are coming from another deep learning toolkit you can start with an [overview for advanced users](https://github.com/Microsoft/CNTK/blob/release/2.1/Tutorials/CNTK_200_GuidedTour.ipynb).

If you have installed CNTK on your machine, after going through the [installation steps](/cognitive-toolkit/Setup-CNTK-on-your-machine),
you can start using CNTK from Python right away (don't forget to ``activate`` your Python environment if you did not install CNTK into your root environment):

```python
    >>> import cntk
    >>> cntk.__version__
    '2.1'
    
    >>> cntk.minus([1, 2, 3], [4, 5, 6]).eval()
    array([-3., -3., -3.], dtype=float32)
```
The above makes use of the CNTK `minus` node with two array constants. Every operator has an `eval()` method that runs a forward 
pass for that node using its inputs, and returns the result. A slightly more interesting example that uses input variables (the 
more common case) is as follows:

```python
    >>> import numpy as np
    >>> x = cntk.input_variable(2)
    >>> y = cntk.input_variable(2)
    >>> x0 = np.asarray([[2., 1.]], dtype=np.float32)
    >>> y0 = np.asarray([[4., 6.]], dtype=np.float32)
    >>> cntk.squared_error(x, y).eval({x:x0, y:y0})
    array([ 29.], dtype=float32)
```

In the above example we are first setting up two input variables with shape `(1, 2)`. We then setup a `squared_error` node with those two variables as 
inputs. Within the `eval()` method we can setup the input-mapping of the data for those two variables. In this case we pass in two numpy arrays. 
The squared error is then of course `(2-4)**2 + (1-6)**2 = 29`.

Most of the data containers like parameters, constants, values, etc. implement
the `asarray()` method, which returns a NumPy interface.

```python
    >>> import cntk as C
    >>> c = C.constant(3, shape=(2,3))
    >>> c.asarray()
    array([[ 3.,  3.,  3.],
           [ 3.,  3.,  3.]], dtype=float32)
    >>> np.ones_like(c.asarray())
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
```

For values that have a sequence axis, `asarray()` cannot work since it requires
the shape to be rectangular and sequences most of the time have different
lengths. In that case, `as_sequences(var)` returns a list of NumPy arrays,
where every NumPy array has the shape of the static axes of `var`.

## Overview and first run

CNTK v2 is a major overhaul of CNTK in that one now has full control over the data and how it is read in, the training and testing loops, and minibatch 
construction. The Python bindings provide direct access to the created network graph, and data can be manipulated outside of the readers not only 
for more powerful and complex networks, but also for interactive Python sessions while a model is being created and debugged.

CNTK v2 also includes a number of ready-to-extend examples and a layers library. The latter allows one to simply build a powerful deep network by 
snapping together building blocks such as convolution layers, recurrent neural net layers (LSTMs, etc.), and fully-connected layers. To begin, we will take a 
look at a standard fully connected deep network in the next section.

### First basic use

The first step in training or running a network in CNTK is to decide which device it should be run on. If you have access to a GPU, training time 
can be vastly improved. To explicitly set the device to GPU, set the target device as follows:

```python
    from cntk.device import try_set_default_device, gpu
    try_set_default_device(gpu(0))
```

Now let's setup a network that will learn a classifier with fully connected layers using only the functions <xref:cntk.layers.higher_order_layers.Sequential>
and <xref:cntk.layers.layers.Dense> from the Layers Library. Create a `simplenet.py` file with the following contents:

```python
from __future__ import print_function
import numpy as np
import cntk as C
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk.logging import ProgressPrinter
from cntk.layers import Dense, Sequential

def generate_random_data(sample_size, feature_dim, num_classes):
     # Create synthetic data using NumPy.
     Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

     # Make sure that the data is separable
     X = (np.random.randn(sample_size, feature_dim) + 3) * (Y + 1)
     X = X.astype(np.float32)
     # converting class 0 into the vector "1 0 0",
     # class 1 into vector "0 1 0", ...
     class_ind = [Y == class_number for class_number in range(num_classes)]
     Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
     return X, Y

def ffnet():
    inputs = 2
    outputs = 2
    layers = 2
    hidden_dimension = 50

    # input variables denoting the features and label data
    features = C.input_variable((inputs), np.float32)
    label = C.input_variable((outputs), np.float32)

    # Instantiate the feedforward classification model
    my_model = Sequential ([
                    Dense(hidden_dimension, activation=C.sigmoid),
                    Dense(outputs)])
    z = my_model(features)

    ce = C.cross_entropy_with_softmax(z, label)
    pe = C.classification_error(z, label)

    # Instantiate the trainer object to drive the model training
    lr_per_minibatch = learning_rate_schedule(0.125, UnitType.minibatch)
    progress_printer = ProgressPrinter(0)
    trainer = C.Trainer(z, (ce, pe), [sgd(z.parameters, lr=lr_per_minibatch)], [progress_printer])

    # Get minibatches of training data and perform model training
    minibatch_size = 25
    num_minibatches_to_train = 1024

    aggregate_loss = 0.0
    for i in range(num_minibatches_to_train):
        train_features, labels = generate_random_data(minibatch_size, inputs, outputs)
        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        trainer.train_minibatch({features : train_features, label : labels})
        sample_count = trainer.previous_minibatch_sample_count
        aggregate_loss += trainer.previous_minibatch_loss_average * sample_count

    last_avg_error = aggregate_loss / trainer.total_number_of_samples_seen

    test_features, test_labels = generate_random_data(minibatch_size, inputs, outputs)
    avg_error = trainer.test_minibatch({features : test_features, label : test_labels})
    print(' error rate on an unseen minibatch: {}'.format(avg_error))
    return last_avg_error, avg_error

np.random.seed(98052)
ffnet()
```

Running `python simplenet.py` (using the correct python environment) will generate this output::

      average      since    average      since      examples
         loss       last     metric       last
      ------------------------------------------------------
        0.693      0.693                                  25
        0.699      0.703                                  75
        0.727      0.747                                 175
        0.706      0.687                                 375
        0.687       0.67                                 775
        0.656      0.626                                1575
         0.59      0.525                                3175
        0.474      0.358                                6375
        0.359      0.245                               12775
         0.29      0.221                               25575
      error rate on an unseen minibatch: 0.0


The example above sets up a 2-layer fully connected deep neural network with 50 hidden dimensions per layer. We first setup two input variables, one for 
the input data and one for the labels. We then call the fully connected classifier network model function which simply sets up the required weights, 
biases, and activation functions for each layer.

We set two root nodes in the network: `ce` is the cross entropy which defines our model's loss function, and `pe` is the classification error. We 
set up a trainer object with the root nodes of the network and a learner. In this case we pass in the standard SGD learner with default parameters and a 
learning rate of 0.02.

Finally, we manually perform the training loop. We run through the data for the specific number of epochs (`num_minibatches_to_train`), get the ``features`` 
and `labels` that will be used during this training step, and call the trainer's `train_minibatch` function which maps the input and label variables that 
we setup previously to the current `features` and `labels` data (numpy arrays) that we are using in this minibatch. We use the convenience function 
`print_training_progress` to display our loss and error every 20 steps and then finally we test our network again using the `trainer` object. It's 
as easy as that!

Now that we've seen some of the basics of setting up and training a network using the CNTK Python API, let's look at a more interesting deep 
learning problem in more detail (for the full example above along with the function to generate random data, please see 
[Tutorials/NumpyInterop/FeedForwardNet.py](https://github.com/Microsoft/CNTK/blob/release/2.1/Tutorials/NumpyInterop/FeedForwardNet.py)).




