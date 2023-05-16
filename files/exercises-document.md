# Episode 01-introduction.md 

## Calculate the output for one neuron
Suppose we have
* Input: X = (0, 0.5, 1)
* Weights: W = (-1, -0.5, 0.5)
* Bias: b = 1
* Activation function _relu_: `f(x) = max(x, 0)`

What is the output of the neuron?

_Note: You can use whatever you like: brain only, pen&paper, Python, Excel..._


## Deep Learning Problems Exercise

Which of the following would you apply Deep Learning to?
1. Recognising whether or not a picture contains a bird.
2. Calculating the median and interquartile range of a dataset.
3. Identifying MRI images of a rare disease when only one or two example images available for training.
4. Identifying people in pictures after being trained only on cats and dogs.
5. Translating English into French.


## Deep Learning workflow exercise

Think about a problem you would like to use Deep Learning to solve.
1. What do you want a Deep Learning system to be able to tell you?
2. What data inputs and outputs will you have?
3. Do you think you will need to train the network or will a pre-trained network be suitable?
4. What data do you have to train with? What preparation will your data need? Consider both the data you are going to predict/classify from and the data you will use to train the network.

Discuss your answers with the group or the person next to you.

## Testing Keras Installation
Lets check you have a suitable version of Keras installed.
Open up a new Jupyter notebook or interactive python console and run the following commands:
~~~
from tensorflow import keras
print(keras.__version__)
~~~

## Testing Seaborn Installation
Lets check you have a suitable version of seaborn installed.
In your Jupyter notebook or interactive python console run the following commands:
~~~
import seaborn
print(seaborn.__version__)
~~~

## Testing Sklearn Installation
Lets check you have a suitable version of sklearn installed.
In your Jupyter notebook or interactive python console run the following commands:
~~~
import sklearn
print(sklearn.__version__)
~~~

# Episode 02-keras.md 

## Penguin Dataset

Inspect the penguins dataset.
1. What are the different features called in the dataframe?
2. Are the target classes of the dataset stored as numbers or strings?
3. How many samples does this dataset have?


## Pairplot

Take a look at the pairplot we created. Consider the following questions:

* Is there any class that is easily distinguishable from the others?
* Which combination of attributes shows the best separation for all 3 class labels at once?


## One-hot encoding vs ordinal encoding

1. How many output neurons will our network have now that we
one-hot encoded the target class?
2. Another encoding method is 'ordinal encoding'.
Here the variable is represented by a single column,
where each category is represented by a different integer
(0, 1, 2 in the case of the 3 penguin species).
How many output neurons will a network have when ordinal encoding is used?
3. (Optional) What would be the advantage of using one-hot versus ordinal encoding
for the task of classifying penguin species?


## Training and Test sets

Take a look at the training and test set we created.
- How many samples do the training and test sets have?
- Are the classes in the training set well balanced?


## Create the neural network

With the code snippets above, we defined a Keras model with 1 hidden layer with
10 neurons and an output layer with 3 neurons.

* How many parameters does the resulting model have?
* What happens to the number of parameters if we increase or decrease the number of neurons
in the hidden layer?


## The Training Curve

Looking at the training curve we have just made.
1. How does the training progress?
* Does the training loss increase or decrease?
* Does it change fast or slowly?
* Is the graph look very jittery?
2. Do you think the resulting trained network will work well on the test set?


## Confusion Matrix

Measure the performance of the neural network you trained and
visualize a confusion matrix.

- Did the neural network perform well on the test set?
- Did you expect this from the training loss you saw?
- What could we do to improve the performance?


# Episode 03-monitor-the-model.md 

## Exercise: Explore the dataset

Let's get a quick idea of the dataset.

* How many data points do we have?
* How many features does the data have (don't count month and date as a feature)?
* What are the different types of measurements (humidity etc.) in the data and how many are there?
* (Optional) Plot the amount of sunshine hours in Basel over the course of a year. Are there any interesting properties that you notice?

date).

3. (Optional) How would we change the model if we would like to output a prediction of the precipitation in Basel in *addition* to the sunshine hours?


## Exercise: Reflecting on our results
* Is the performance of the model as you expected (or better/worse)?
* Is there a noteable difference between training set and test set? And if so, any idea why?
* (Optional) When developing a model, you will often vary different aspects of your model like
which features you use, model parameters and architecture. It is important to settle on a
single-number evaluation metric to compare your models.
* What single-number evaluation metric would you choose here and why?


## Exercise: Baseline
1. Looking at this baseline: Would you consider this a simple or a hard problem to solve?
2. (Optional) Can you think of other baselines?


## Exercise: plot the training progress.

1. Is there a difference between the training and validation data? And if so, what would this imply?
2. (Optional) Take a pen and paper, draw the perfect training and validation curves.
(This may seem trivial, but it will trigger you to think about what you actually would like to see)


## Exercise: Try to reduce the degree of overfitting by lowering the number of parameters

We can keep the network architecture unchanged (2 dense layers + a one-node output layer) and only play with the number of nodes per layer.
Try to lower the number of nodes in one or both of the two dense layers and observe the changes to the training and validation losses.
If time is short: Suggestion is to run one network with only 10 and 5 nodes in the first and second layer.

* Is it possible to get rid of overfitting this way?
* Does the overall performance suffer or does it mostly stay the same?
* How low can you go with the number of parameters without notable effect on the performance on the validation set?




## Open question: What could be next steps to further improve the model?

With unlimited options to modify the model architecture or to play with the training parameters, deep learning can trigger very extensive hunting for better and better results.
Usually models are "well behaving" in the sense that small chances to the architectures also only result in small changes of the performance (if any).
It is often tempting to hunt for some magical settings that will lead to much better results. But do those settings exist?
Applying common sense is often a good first step to make a guess of how much better *could* results be.
In the present case we might certainly not expect to be able to reliably predict sunshine hours for the next day with 5-10 minute precision.
But how much better our model could be exactly, often remains difficult to answer.

* What changes to the model architecture might make sense to explore?
* Ignoring changes to the model architecture, what might notably improve the prediction quality?


# Episode 04-advanced-layer-types.md 

## Explore the data

Familiarize yourself with the CIFAR10 dataset. To start, consider the following questions:
1. What is the dimension of a single data point? What do you think the dimensions mean?
2. What is the range of values that your input data takes?
3. What is the shape of the labels, and how many labels do we have?
4. (Optional) We are going to build a new architecture from scratch to get you
familiar with the convolutional neural network basics.
But in the real world you wouldn't do that.
So the challenge is: Browse the web for (more) existing architectures or pre-trained models that are likely to work
well on this type of data. Try to understand why they work well for this type of data.


## Number of parameters

Suppose we create a single Dense (fully connected) layer with 100 hidden units that connect to the input pixels, how many parameters does this layer have?


## Border pixels

What, do you think, happens to the border pixels when applying a convolution?


## Number of model parameters

Suppose we apply a convolutional layer with 100 kernels of size 3 * 3 * 3 (the last dimension applies to the rgb channels) to our images of 32 * 32 * 3 pixels. How many parameters do we have? Assume, for simplicity, that the kernels do not use bias terms. Compare this to the answer of the previous exercise


## Convolutional Neural Network

Inspect the network above:
* What do you think is the function of the `Flatten` layer?
* Which layer has the most parameters? Do you find this intuitive?
* (optional) Pick a model from https://paperswithcode.com/sota/image-classification-on-cifar-10 . Try to understand how it works.


## Network depth

What, do you think, will be the effect of adding a convolutional layer to your model? Will this model have more or fewer parameters?
Try it out. Create a `model` that has an additional `Conv2d` layer with 50 filters after the last MaxPooling2D layer. Train it for 20 epochs and plot the results.

**HINT**:
The model definition that we used previously needs to be adjusted as follows:
~~~
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
# Add your extra layer here
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(50, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)
~~~

## Why and when to use convolutional neural networks

1. Would it make sense to train a convolutional neural network (CNN) on the penguins dataset and why?
2. Would it make sense to train a CNN on the weather dataset and why?
3. (Optional) Can you think of a different machine learning task that would benefit from a
CNN architecture?


## Vary dropout rate

1. What do you think would happen if you lower the dropout rate? Try it out, and
see how it affects the model training.
2. You are varying the dropout rate and checking its effect on the model performance,
what is the term associated to this procedure?



