![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document 2023-05-23-ds-dl-intro day 2

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: https://tinyurl.com/dl-intro-day2

Collaborative Document day 1: https://tinyurl.com/dl-intro-day1

Collaborative Document day 2: https://tinyurl.com/dl-intro-day2

## 👮Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

## 🖥 Workshop website
https://esciencecenter-digital-skills.github.io/2023-05-23-ds-dl-intro/

🛠 Setup
https://esciencecenter-digital-skills.github.io/2023-05-23-ds-dl-intro/#setup

## 👩‍🏫👩‍💻🎓 Instructors

Sven van der Burg, Olga Lyashevska

## 🧑‍🙋 Helpers

Eva Viviani, Flavio Hafner

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city


## 🗓️ Agenda
09:30	Welcome and recap
09:45	Monitor the training processs
10:30	Break
10:40	Monitor the training process
11:30	Break
11:40	Monitor the training process
12:30	Lunch Break
13:30	Advanced layer types
14:30	Break
14:40	Advanced layer types
15:30	Break
15:40	Advanced layer types
16:15	Post-workshop Survey, recap + questions
16:30	Drinks

## 🏢 Location logistics
* Coffee and toilets are in the hallway, just outside of the classroom.
* If you leave the building, 
  be sure to be accompanied by someone from the escience center to let you back in through the groundfloor door
* For access to this floor you might need to ring the doorbell so someone can let you in
* In case of an emergency, you can exit our floor using the main staircase.
  Or follow green light signs at the ceiling to the emergency staircase.
* **Wifi**: Eduroam should work. Otherwise use the 'matrixbuilding' network, password should be printed out and available somewhere in the room.

## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl.

## 🔧 Exercises

#### Architecture of the network

As we want to design a neural network architecture for a regression task, see if you can first come up with the answers to the following questions:

1. What must be the dimension of our input layer?

2. We want to output the prediction of a single number. The output layer of the `NN` hence cannot be the same as for the classification task earlier. This is because the `softmax` activation being used had a concrete meaning with respect to the class labels which is not needed here. What output layer design would you choose for regression? Hint: A layer with `relu` activation, with `sigmoid` activation or no activation at all?

3. (Optional) How would we change the model if we would like to output a prediction of the precipitation in Basel in addition to the sunshine hours?


### Reflecting on our results
1. Is the performance of the model as you expected (or better/worse)?
2. Is there a noteable difference between training set and test set? And if so, any idea why?
3. (Optional) When developing a model, you will often vary different aspects of your model like
which features you use, model parameters and architecture. It is important to settle on a
single-number evaluation metric to compare your models.
4. What single-number evaluation metric would you choose here and why?


### Looking at this baseline: 

1. Would you consider this a simple or a hard problem to solve?
2. (Optional) Can you think of other baselines?


### plot the training progress.
1. Is there a difference between the training and validation data? And if so, what would this imply?
2. (Optional) Take a pen and paper, draw the perfect training and validation curves. (This may seem trivial, but it will trigger you to think about what you actually would like to see)



### Try to reduce the degree of overfitting by lowering the number of parameters
We can keep the network architecture unchanged (2 dense layers + a one-node output layer) and only play with the number of nodes per layer. Try to lower the number of nodes in one or both of the two dense layers and observe the changes to the training and validation losses. If time is short: Suggestion is to run one network with only 10 and 5 nodes in the first and second layer.

1. Is it possible to get rid of overfitting this way?
2. Does the overall performance suffer or does it mostly stay the same?
3. How low can you go with the number of parameters without notable effect on the performance on the validation set?


TIP: modify the `create_nn()` function to accept number of nodes for the dense layers as arguments. So that you could potentially use it as: `create_nn(nodes1=100, nodes2=50)`

## Open question: What could be next steps to further improve the model?

With unlimited options to modify the model architecture or to play with the training parameters, deep learning can trigger very extensive hunting for better and better results.
Usually models are "well behaving" in the sense that small chances to the architectures also only result in small changes of the performance (if any).
It is often tempting to hunt for some magical settings that will lead to much better results. But do those settings exist?
Applying common sense is often a good first step to make a guess of how much better *could* results be.
In the present case we might certainly not expect to be able to reliably predict sunshine hours for the next day with 5-10 minute precision.
But how much better our model could be exactly, often remains difficult to answer.

* What changes to the model architecture might make sense to explore?
* Ignoring changes to the model architecture, what might notably improve the prediction quality?

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


## Exercise: understanding convolutional neural networks
### 1. Border pixels

What, do you think, happens to the border pixels when applying a convolution?


### 2. Number of model parameters

Suppose we apply a convolutional layer with 100 kernels of size 3 * 3 * 3 (the last dimension applies to the rgb channels) to our images of 32 * 32 * 3 pixels. How many parameters do we have? Assume, for simplicity, that the kernels do not use bias terms. Compare this to the answer of the previous exercise


### 3. Convolutional Neural Network
So let us look at a network with a few convolutional layers. We need to finish with a Dense layer to connect the output cells of the convolutional layer to the outputs for our classes.
```python
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model_small")

model.summary()
```
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

## Recap exercise
Think about what you learned these 2 days
* What questions do you still have?
* Are there are any incremental improvements that can benefit your projects?
* What’s nice that we learnt but is overkill for your current work?

## 🧠 Collaborative Notes

Goal for today's dataset: Predict number of sunshine hours on the next day

Yesterday we trained a dense neural network on a classification task (i.e., classifying penguins). For this one, hot encoding was used together with a Categorical Crossentropy loss function. This measured how close the distribution of the neural network outputs corresponds to the distribution of the three values in the one hot encoding. Today we want to work on a regression task, thus not predicting a class label (or integer number) for a datapoint. In regression, we like to predict one (and sometimes many) values of a feature. This is typically a floating point number.

We have loaded our `weather` dataset and we have plotted the sunshine hours by day with the following code:

```python
# load data
data = pd.read_csv("https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1")

# plot sunshine hours
data.iloc[:365]['BASEL_sunshine'].plot(xlabel="Day",ylabel="Basel sunshine hours")

```

#### 3. Prepare the data
We will subset the data to focus only the first 3 years. By inspecting the data, you can see that the column `DATE` is ordered and numbered, so you can subset it based on number of rows. 

We subset the data and save it into `X_data` and `y_data`

```python
nr_rows = 365 * 3
X_data = data.loc[:nr_rows]
X_data = X_data.drop(columns = ['DATE', 'MONTH']) # features

y_data = data.loc[1:(nr_rows + 1)]["BASEL_sunshine"] # labels
```

Let's import `train_test_split` function from sklearn. Then we create the training set and leave the remainder of 30 % of the data to the two hold-out sets. 

```python
from sklearn.model_selection import train_test_split

X_train, X_holdhout, y_train, y_holdout = train_test_split(X_data, y_data, test_size=0.3, random_state=0)

```

Afterwards, we split the 30 % of the data in two equal sized parts.
```python
X_val, X_test, y_val, y_test = train_test_split(X_holdhout, y_holdout, test_size=0.5, random_state=0)
```

#### 4. Choose a pretrained model or start building architecture from scratch

Let's google `state of the art weather prediction keras`. Look at the documentation on keras's website, and look at whether the data in the example is similar to what we have in our dataset. Let's see what type of model they have. They use a `LSTM` layer. This is a particular type of architecture that is used for time series data. We can try to follow what Keras' team suggests and continue from there.

However, in our problem we need to predict the next value on the next day, so we don't need really time-series. However, there are many approaches around problems, so it's always useful to see other examples that you can find out there.

Exercise time now: Architecture of the network

Answer to Q1: We got 89 features, so we need 89 input neurons
Answer to Q2: We want to output only one number. So we'll go for 1 output neuron with no activation function (or linear activation function)

Q -- why the `relu` function?
A -- Because of the type of dataset we got, we could have distance between points that would go flat, and this would impede prediction. `relu` function is able to overcome this characteristics in the data. However, in general one needs to try it out as well -- by trial and error we can learn about our data and see which function is best suited to our data. You start with an intuition, then you refine it later on.

#### create the NN

```python
from tensorflow import keras

def create_nn():
    # input layer
    inputs = keras.Input(shape=(X_data.shape[1], ), name='input') #input layer
    
    # dense layers
    layers_dense = keras.layers.Dense(100, 'relu')(inputs) # Dense layers 1
    layers_dense = keras.layers.Dense(50, 'relu')(layers_dense) # Dense layers 2
    
    # output layer
    outputs = keras.layers.Dense(1)(layers_dense) # 1 output neuron
    
    return keras.Model(inputs=inputs, outputs=outputs, name='weather_prediction_model')
```

Why using 100 or 50 neurons? these decisions are arbitrary. Many of these decisions have to be taken by consulting previous research on the subject, and by inspecting your data. However, ultimately is also by trial and error. You start with an educated guess, and refine it by experimenting yourself.

Create the model:
```python
model = create_nn()
```

Overview of the model:

```python
model.summary()
```

This time we got over 14.000 parameters. Note also that the dense layers have been renamed by Keras automatically as `dense_1` and `dense_2`.


#### 5. Choose a loss function and optimizer
Q: what loss function would you use in this case?

This is a regression problem, so we can use `MSE` as we discussed yesterday.

We could use it that way
```python
model.compile(loss="mse")
```
but we need also an optimizer. Let's use again adam like yesterday. We can also ask Keras to give us back as output a RootMeanSquaredError

```python
model.compile(loss="mse", optimizer="adam", metrics=[keras.metrics.RootMeanSquaredError()])
```

We are going to re-use this code several times, so let's wrap it into a function:

```python
def compile_model(model):
    model.compile(loss="mse", 
                  optimizer="adam", 
                  metrics=[keras.metrics.RootMeanSquaredError()])
    
```
Let's use it now:

```python
compile_model(model)
```

#### 6. Train the model

What did we use yesterday? What function? It's model.`fit`

We also add `batch size`. It's usually the ^2. At the beginning we don't know what is the batch size, this is just the part of the dataset that will be dedicated to training. If the size is larger, the model will learn more about your data, but it may be computationally challenging. If it's smaller, then there might be not enough data, while however this won't be computationally intense. It's again another trial and error.


```python
history = model.fit(X_train, y_train,
             batch_size=32,
             epochs=200,
             verbose=0)
```

We have now trained our second model. Congrats!

Let's look at the output -- how did the model learn? for this purpose, we'll write another function so that we can reuse the code.

```python
import seaborn as sns
import matplotlib.pyplot as plt

def plot_history(metrics):
    # let's convert the ouput of history (arrays) into a pandas dataframe
    
    history_df = pd.DataFrame.from_dict(history.history)
    sns.lineplot(data=history_df[metrics])
    plt.xlabel("epochs")
    plt.ylabel("RMSE")
```

Let's use it now to visualise our model history:

```python
plot_history("root_mean_squared_error")
```

#### 7. Perform prediction / regression
Let's see now how the model trained fare on unseen data:

```python
y_train_predicted = model.predict(X_train)
y_test_predicted = model.predict(X_test)
```

#### 8. Measure performance

There are many ways to measure performance. For a classification task yesterday we saw the confusion matrix. For a regression task, it makes more sense to compare true and predicted values in a scatter plot.

```python
def plot_predictions(y_pred, y_true, title):
    plt.scatter(y_pred, y_true, s=10, alpha=0.5)
    plt.xlabel("Predicted sunshine hours")
    plt.ylabel("True sunshine hours")
    plt.title(title)
```

Now we can try to use it:

```python
plot_predictions(y_train_predicted, y_train, title='Predictions on the training set')
```

Let's do the same with test:

```python
plot_predictions(y_test_predicted, y_test, title='Predictions on the test set')
```

How is our performance? performance on the train set seems reasonable, but not on test. Why?

Let's evaluate our model:

```python
train_metrics = model.evaluate(X_train, y_train, return_dict=True)

test_metrics = model.evaluate(X_test, y_test, return_dict=True)
```

Inspect train_metrics:

```python
train_metrics
test_metrics
```

If you look at the loss function output this is very small in the training, but this value is much higher in test. This information, together with the plots above suggest us that our model overfitted. As a result, it makes much more accurate predictions on the training data than on unseen test data.

This is a common problem in ML and DL, what could have caused it? In normal ML, this might be due to having too many parameters. Reducing the number of parameters saves you from overfitting. However, for DL is different.

It could be to have too many parameters, which in DL are layers and/or nodes. Later on today we'll see a special tool/technique to exploit regularisation and see how to avoid overfitting.

#### 9. Tune hyperparameters

Before going deeply into tuning hyperparameters, let's define our problem: How well must a model perform before we consider it a good model?

Our problem is to predict the next day sunshine hours. We assume that the sunshine hours of day + 1 are going to be v. similar to today.

Let's put aside 'BASEL_sunshine' -- this contains the sunshine hours from one day before what we have as a label.

```python
y_baseline_prediction = X_test['BASEL_sunshine']
plot_predictions(y_baseline_prediction, y_test, tile="Baseline predictions on the test set")
```

By inspecting the plot it's unclear how our model fares. It doesn't look great, so let's look at the RMSE

```python
from sklearn.metrics import mean_squared_error

rmse_baseline = mean_squared_error(y_test, y_baseline_prediction, squared=False)

```
The reason for having `squared=False` it's because we are interested in obtaining the root without having it squared.

Inspect the outputs:
```python
rmse_baseline
test_metrics['root_mean_squared_error']
```

[Exercise](https://codimd.carpentries.org/X3uT-aUFT0OXEANzEjQiJw?both#Looking-at-this-baseline) now.

A1: it may remain hard to find a good algorithm which exceeds our baseline by orders of magnitude.
A2: Another possible baseline, would be to take the past 2 days. Maybe we can have the rolling average of the past 7 days?

Baselines are essential in DL/ML. You won't be able to know how good is your model doing without it.

#### Test vs validation

What is a validation set? Why would you use it? How is different from test?

In the ML/DL domain, researchers use this terminology always differently. Be sure to know what they mean when they talk about validation/train/test. 

Validation set is used to fine-tune parameters after training. Test instead is used always to evaluate the performance.

Good practice is to have 3 (test/train/validation).

```python
model = create_nn()
```

Let's refresh our memory about the model architecture:

```python
??create_nn
```

Now we want to compile the model with our custom-made function:

```python
compile_model(model)
```

But now we train it with the small addition of also passing it our validation set:

```python
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=200,
                    validation_data=(X_val, y_val))
```

Let's inspect the history output. With this, we can plot both the performance on the training data and on the validation data:


```python
plot_history(['root_mean_squared_error', 'val_root_mean_squared_error'])
```

[Exercise](https://codimd.carpentries.org/X3uT-aUFT0OXEANzEjQiJw?both#plot-the-training-progress) now.

A1: The model predictions on the validation set quickly seem to reach a plateau while the performance on the training set keeps improving. That is a common signature of overfitting.

A2: Ideally you would like the training and validation curves to be identical and slope down steeply to 0.

### Counteract model overfitting

[Exercise](https://codimd.carpentries.org/X3uT-aUFT0OXEANzEjQiJw?both#Try-to-reduce-the-degree-of-overfitting-by-lowering-the-number-of-parameters) now.

TIP: modify the `create_nn()` function to accept number of nodes for the dense layers as arguments. So that you could potentially use it as: `create_nn(nodes1=100, nodes2=50)`

A1: 

First, let's tweak the create_nn function this way:

```python
def create_nn(nodes1=100, nodes2=50):
    # Input layer
    inputs = keras.layers.Input(shape=(X_data.shape[1],), name='input')

    # Dense layers
    layers_dense = keras.layers.Dense(nodes1, 'relu')(inputs)
    layers_dense = keras.layers.Dense(nodes2, 'relu')(layers_dense)

    # Output layer
    outputs = keras.layers.Dense(1)(layers_dense)

    return keras.Model(inputs=inputs, outputs=outputs, name="model_small")
```

Then, we run it:

```python
model = create_nn(nodes1=10, nodes2=5)
model.summary()
```

We have reduced the number of parameters from over 14.000 to over 900.


A2: 

Let's compile and fit the model.
```python
compile_model(model)
history = model.fit(X_train, y_train,
                    batch_size = 32,
                    epochs = 200,
                    validation_data=(X_val, y_val))
```
Then we visualise it with our custom made function
```python
plot_history(['root_mean_squared_error', 'val_root_mean_squared_error'])
```

We saw that reducing the number of parameters can be a strategy to avoid overfitting. It worked somehow for us, but as you can see, this requires a bit of trial and error. This might be time consuming in the long run. There is a way however to automatically **stop** when the fit looks alright. How do we do that? via Early stopping. Another technique is Batch normalisation (BatchNorm).


#### Early stopping
What is early stopping? It's a technique by which you stop the model training if things do not seem to improve any more.

```python
model = create_nn()
compile_model(model)
```

```python
from tensorflow.keras.callbacks import EarlyStopping

earlystopper = EarlyStopping(monitor="val_loss", patience=10)

history = model.fit(X_train, y_train,
                    batch_size = 32,
                    epochs = 200,
                    validation_data=(X_val, y_val),
                    callbacks=[earlystopper])

plot_history(['root_mean_squared_error', 'val_root_mean_squared_error'])
```

#### Batch norm: the standard scaler for DL

The main idea behind it is that you normalise your inputs. We re-use our `create_nn` function and adapt it to have now a BatchNormalization step.

```python
from tensorflow.keras.layers import BatchNormalization


def create_nn(nodes1=100, nodes2=50):
    # Input layer
    inputs = keras.layers.Input(shape=(X_data.shape[1],), name='input')

    # Dense layers
    layers_dense = keras.layers.BatchNormalization()(inputs) # normalisation layer here
    layers_dense = keras.layers.Dense(nodes1, 'relu')(layers_dense)
    layers_dense = keras.layers.Dense(nodes2, 'relu')(layers_dense)

    # Output layer
    outputs = keras.layers.Dense(1)(layers_dense)

    # Defining the model and compiling it
    return keras.Model(inputs=inputs, outputs=outputs, name="model_batchnorm")

```

```python
model = create_nn()
compile_model(model)
model.summary()
```

A way to improve prediction quality could be data augmentation. We can also change the architecture. Try to be creative based on the type of problem you have.

# Advanced layer types
We're going to load a new dataset. This is the `CIFAR10 dataset`.

```python
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
```

#### Goal: From 10 classes, predict/classify the one that is within the image

```python
n=5000
train_images = train_images[:n]
train_labels = train_labels[:n]
```

[Exercise](https://codimd.carpentries.org/X3uT-aUFT0OXEANzEjQiJw?both#Explore-the-data) now.

A1:

```python
train_images.shape
```

`5000` is the number of training images that we have selected. The remainder of the shape, namely `32, 32, 3`, denotes the dimension of one image. The last value `3` is typical for color images, and stands for the three color channels Red, Green, Blue.

A2:
```python
train_images.min(), train_images.max()
```

A3:
```python
train_labels.shape
train_labels.min(), train_labels.max()
```

```python
import numpy as np
np.unique(train_labels)
```

#### 3. Prepare the data
let's get rid of the weird range of the `train_images` and `test_images`. We do that by typing:

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

#### 4. Choose a pretrained model or build from scratch?

You must decide based on the info available out there. For example, for the `cifar10` dataset, you can google `cifar10 state of the art keras` and you get an [example](https://paperswithcode.com/sota/image-classification-on-cifar-10), or this [other example](https://github.com/Adeel-Intizar/CIFAR-10-State-of-the-art-Model).


```python
dim = train_images.shape[1] * train_images.shape[2] * train_images.shape[3]
```

Let's say we have 100 hidden units, how many parameters do we have?


```python
(100 * dim) + 100 # the last 100 is the bias
```

The red cell in the right side of the image is the result of the matrix multiplication done on the left side. Note that we get:

100 * -1 | 100 * -1 | 100 * -1
0 | 0 | 0
0 | 0 | 0

by summing the results of the multiplications we get: -300 (highlighted in red on the rightmost matrix)

![](https://codimd.carpentries.org/uploads/upload_74fd239691b059cbe10646dc09f36d53.png)

in a convolutional layer, our hidden units are the number of matrices (or kernels), where the values contained in the matrices are the weights learned by the model during training.

[Exercise](https://codimd.carpentries.org/X3uT-aUFT0OXEANzEjQiJw?both#Exercise-understanding-convolutional-neural-networks) now.

A1: So what does happen to the border pixel when we apply a convolution? In this case, it will ‘pad’ the borders, this is indeed called "padding"

A2: number of model parameters? We have 100 matrices with 3 * 3 * 3 = 27 values each so that gives 27 * 100 = 2700 weights.

All these kernels are pre-trained by the NN. 


A3: So what does the flatten layer does? The flatten layer converts the 28x28x50 output of the convolutional layer into a single one-dimensional vector, that can be used as input for a dense layer.

A4: Which layer has the most parameters? The last dense layer. That is because this layer connects to the 10 output classes. This increases the number of parameters, i.e., the number of connections.


#### Add a pooling layer

```python
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)

# add pooling layer here
x = keras.layers.MaxPooling2D((2, 2))(x)

x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)

# add another pooling layer here
x = keras.layers.MaxPooling2D((2, 2))(x)

x = keras.layers.Flatten()(x)
x = keras.layers.Dense(50, activation='relu')(x) # a new dense layer

outputs = keras.layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model_small")

model.summary()
```

So what did it happen here? pooling reduces drastically the number of parameters. This prevents overfitting. So pooling directly alters the number of dimensions of the image and reduce it of a scaling factor -- as if we were decreasing the resolution of our image. The rationale behind this is to avoid overfitting, by allowing the model to focus on the high-level features.

#### Choose a loss function and optimiser

Let's compile the model and choose loss/optimiser. Adam always works, but for the loss, we need to think about what is our dependent variable. It is not ordinal, but still continuous. This is called `SparseCategoricalCrossentropy`.

```python
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

Any other metrics we could use? precision, recall, f1 score, etc.


```python
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))
```

```python
import seaborn as sns
import pandas as pd

history_df = pd.DataFrame.from_dict(history.history)
sns.lineplot(data=history_df[['accuracy', 'val_accuracy']])
```

```python
sns.lineplot(data=history_df[['loss', 'val_loss']])
```

Looking at these plots we can see that the lines develop in opposite ways in the two plots.

[Exercise](https://codimd.carpentries.org/X3uT-aUFT0OXEANzEjQiJw?both#Network-depth) now.

A1:
```python
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(50, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model")
```

```python
model.summary()
```

```python
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

history_df = pd.DataFrame.from_dict(history.history)
sns.lineplot(data=history_df[['accuracy', 'val_accuracy']])
```

#### Dropout layers
It basically drops a number of nodes, and for each epoch it will be a different set, that is to give all nodes a chance to observe enough training data to learn its weights.

![](https://codimd.carpentries.org/uploads/upload_c6480913c2dfcca2c532b45e378f2d91.png)

You simply add this layer to the above code (note that here we are dropping 20% of the input units):

```python
x = keras.layers.Dropout(0.8)(x) 
```

The code will ultimately look like that:
```python
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.Dropout(0.8)(x) # <----- This is new!!!! 
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(50, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)

model_dropout = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model")

model_dropout.summary()
```
Q1: Would it make sense to train a CNN on the penguin dataset?
A1: There is no need to train a CNN on the penguin dataset, as this makes sense to use it with images. 

Q2: Would it make sense to train a CNN on the weather dataset? 
A2: Not really if you wish to predict the sunshine hours (as we did before). If we were intersted in the temporal relationship over days, then CNN would have been useful. 

Q3: Can you think of a different ML task that would benefit from a CNN architecture?
A3:
- Text data
- Time series, specifically audio
- Molecular structures


# Feedback

## ---> Fill in the post-workshop survey!!!! <---
THIS ONE
https://www.surveymonkey.com/r/M6H9KVM



----
Write down one thing that went well and that could be improved. think about the content, the pace, the instructors, the coffee, …

### What went well? :smile_cat: 
* It was nice that we got to practice/experiment more by ourselves 
* The last exercise was great. Also the discussions regarding the overfitting problem were quite useful. 
* Nice ways to bring up problems related to risks of overfitting
* Liked the last exercise to get a feeling of training and retraining and thinking about the created model and how to improve. 
* I really liked the last exercise where we adjusted the number of nodes. Having such longer exercises where we go over the code again ourselves is really helpful and I learn a lot from this. 
* like the explanations and the thinking about if the result is good and how to improve 
* Good that we are practising more ourselves
* Good to get insights into model performance on training & validation data
* interesting hands on
* Really liked the overfitting exercise
* liked getting to play with the code myself a bit (could have been a bit more)
* most things went well, overall very good impression of the course. Thank you for all your effort!



### What can be improved? :fire:
* The pace was relatively slow, we don't go deeper in on specific issues of deep learning that are different from modeling in general.
* still a bit too slow for my taste
* Lots of typing to do. One mistake and you are way behind.
    * please note that everything that is typed live is also typed in the collaborative notes. Just copy and paste the code if you think you made a mistake.
* Sometimes we glance over things that seem to be relevant. For example: how do you determine the number of layers? And the number of nodes? Does the number of nodes in the first layer always > the number of nodes in the second layer? These things seem very relevant and I miss some more explanation there (relative to going over all lines of code one-by-one).
* Still a little bit slow, but I get that we all have different preferences regarding the pace
* More discussion on how can we improve?
* Getting some sort of a state-of-the-art would be good, and getting an overview of other network types 

### Sven's summary
* You like the more advanced exercises, we will get more of those
* We should go faster and slower. How about slower typing, more copy-pasting faster discussions?
* We will have time to discuss more on how to improve the model at the end of part 3.
* We make more time for questions and you promise to ask questions when you have them :),,

## 📚 Resources

- [Kaggle](https://www.kaggle.com/)
- [Kaggle competitions](https://www.kaggle.com/competitions/)
- What are [non-trainable parameters](https://stackoverflow.com/questions/47312219/what-is-the-definition-of-a-non-trainable-parameter)?
- [Transformers](https://huggingface.co/docs/transformers/index)
- https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00558-4
- https://github.com/matchms/ms2deepscore/blob/main/ms2deepscore/models/SiameseModel.py
- https://huggingface.co/docs/transformers/v4.29.1/en/model_doc/gpt-sw3

- [Tensorboard](https://www.tensorflow.org/tensorboard)

- [Weights and biases](https://wandb.ai/site)
- [2-min-papers](https://www.youtube.com/channel/UCbfYPyITQ-7l4upoX8nvctg)
- [Fast.ai course](https://course.fast.ai/)