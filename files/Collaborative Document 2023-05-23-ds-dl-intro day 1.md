![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document 2023-05-23-ds-dl-intro day 1

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: https://tinyurl.com/dl-intro-day1

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
09:30	Welcome and icebreaker
09:45	Introduction to Deep Learning
10:30	Break
10:40	Introduction to Deep Learning
11:30	Break
11:40	Classification by a Neural Network using Keras
12:30	Lunch Break
13:30	Classification by a Neural Network using Keras
14:30	Break
14:40	Classification by a Neural Network using Keras
15:30	Break
15:40	Monitor the training processs
16:15	Wrap-up
16:30	END

## 🏢 Location logistics
* Coffee and toilets are in the hallway, just outside of the classroom.
* If you leave the building, 
  be sure to be accompanied by someone from the escience center to let you back in through the groundfloor door
* For access to this floor you might need to ring the doorbell so someone can let you in
* In case of an emergency, you can exit our floor using the main staircase.
  Or follow green light signs at the ceiling to the emergency staircase.
* **Wifi**: Eduroam should work. Otherwise use the 'matrixbuilding' network, password should be printed out and available somewhere in the room.

## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .



## 🔧 Exercises

### Calculate output for one neuron

Suppose we have

Input: X = (0, 0.5, 1)
Weights: W = (-1, -0.5, 0.5)
Bias: b = 1
Activation function relu: f(x) = max(x, 0)
What is the output of the neuron?

Note: You can use whatever you like: brain only, pen&paper, Python, Excel…


**you can write your solutions below, next to your name**





Relu function
- introduces non-linearity in the network
- is mathematically very simple -- for an input $x$, it returns the same input if $x>0$, or $0$ otherwise. There is not much computation needed.
- this function works very well in many neural networks



### Exercise: Mean Squared Error
One of the simplest loss functions is the Mean Squared Error. MSE = $\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$ .
It is the mean of all squared errors, where the error is the difference between the predicted and expected value.
In the following table, fill in the missing values in the 'squared error' column. What is the MSE loss for the predictions on these 4 samples?
<table>
<thead>
<tr>
<th>Prediction</th> <th>Expected value</th> <th>Squared error</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>              <td>-1</td>                 <td>4</td>
</tr>
<tr>
<td>2</td>             <td>-1</td>                   <td>..</td>
</tr>
<tr>
<td>0</td>             <td>0</td>                    <td>..</td>
</tr>
<tr>
<td>3</td>             <td>2</td>                    <td>..</td>
</tr>
<tr>
<td></td>             <td>MSE:</td>              <td>..</td>
</tr>
</tbody>
</table>


**you can write your solutions below, next to your name**




There are many different loss functions that one can use. The trick is to choose the one that makes most sense given the structure of the data and the prediction problem/context of what the prediction is used for. 


### Deep Learning Problems Exercise

Which of the following would you apply Deep Learning to?
1. Recognising whether or not a picture contains a bird.
    - yes, image classification
3. Calculating the median and interquartile range of a dataset.
    - not necessary, can use simple maths
5. Identifying MRI images of a rare disease when only one or two example images available for training.
    - not enough training data
6. Identifying people in pictures after being trained only on cats and dogs.
    - training data does not fit application domain
7. Translating English into French.
    - yes

**write your solutions below, next to your name**



### Penguins dataset exercise


Inspect the penguins dataset.
1. What are the different features called in the dataframe?
2. Are the target classes of the dataset stored as numbers or strings?
3. How many samples does this dataset have?
4. (optional) Do you think this will be an easy or difficult machine learning problem?
5. (optional) Do you think we have enough samples in our dataset to train our neurla network?
6. (optional) Are there missing values and do you think it will be a problem when training our network?

**write your solutions below, below your name**


**solutions**

1. What are the different features called in the dataframe?
```python
penguins.columns
```

2. Are the target classes of the dataset stored as numbers or strings?
    
```python
penguins["species"]
# `dtype: object` is string type in pandas

```

3. How many samples does this dataset have?
```python
len(penguins)
```


### Pairplot exercise

Take a look at the pairplot we created. Consider the following questions:

* Is there any class that is easily distinguishable from the others?
* Which combination of attributes shows the best separation for all 3 class labels at once?

Optional
- what happened to the sex and island featueres, why are they not shown and can you find a way to visualize them?
- can you answer the above questions for those 2 features?


**write your solutions below, below to your name**


#### Solution

1. Is there any class that is easily distinguishable from the others?
- green dots, Gentoo species


2. Which combination of attributes shows the best separation for all 3 class labels at once?
- flipper and bill length
- also other combinations are possible, ie  bill length and bill depth



### One-hot encoding vs ordinal encoding

1. How many output neurons will our network have now that we one-hot encoded the target class?
2. Another encoding method is 'ordinal encoding'.Here the variable is represented by a single column, where each category is represented by a different integer (0, 1, 2 in the case of the 3 penguin species). How many output neurons will a network have when ordinal encoding is used?

Optional
- What would be the advantage of using one-hot versus ordinal encoding for the task of classifying penguin species?
- Let's say we want to use the sex and island columns as features. How would you represent them numerically? What will be the size of our feature vector in that case?



#### Solution

1. How many output neurons will our network have now that we one-hot encoded the target class?
3 neurons: one neuron for each class


2. Another encoding method is 'ordinal encoding'.Here the variable is represented by a single column, where each category is represented by a different integer (0, 1, 2 in the case of the 3 penguin species). How many output neurons will a network have when ordinal encoding is used?
1 neuron

Ordinal encoding? -- this implies a numerical ordering between the classes, which is not appropriate in the present case.


### Training and Test sets

Take a look at the training and test set we created.
- How many samples do the training and test sets have?
- Are the classes in the training set well balanced?

Optional
- let's say that the balance in the dataset will be a big problem, how would you solve it?
- Could you also use cross-validation? Why would you want to use it?




**Solution**
How many samples do the training and test sets have?
```python
display(X_train.shape)
display(X_test.shape)
```


Are the classes in the training set well balanced?

```python
y_train.sum()
```

### Create the neural network

With the code snippets above, we defined a Keras model with 1 hidden layer with
10 neurons and an output layer with 3 neurons.

* How many parameters does the resulting model have?
* What happens to the number of parameters if we increase or decrease the number of neurons in the hidden layer?

Optional
- Have a look at https://modelzoo.co/ to see if you can find good existing architectures that fit the problem.




#### Solutions

1. How many parameters does the resulting model have?
83. see the summary of the model.

2. What happens to the number of parameters if we increase or decrease the number of neurons in the hidden layer?
increase/decrease. 8 per hidden layer: 4 parameters for the input neurons, one bias, plus the 3 connections to the output layer

3. Have a look at https://modelzoo.co/ to see if you can find good existing architectures that fit the problem.
for this basic model, there is no single option for pre-trained model.


### The Training Curve

Looking at the training curve we have just made.
1. How does the training progress?
* Does the training loss increase or decrease?
* Does it change fast or slowly?
* Is the graph look very jittery?
2. Do you think the resulting trained network will work well on the test set?




#### Solutions

1. How does the training progress?
- it looks fine: it is decreasing with more epochs

2. Do you think the resulting trained network will work well on the test set?
- we don't know from this figure!
- we'd want to look at the error in the *test* data, not the *training* data
- in short, we don't know the best number of epochs. there are techniques and tools to help us choose the best moment to stop.
    - for instance, early stopping: stops training as soon as some pre-defined conditions are met (ie, prediction does not get better for 10 epochs)



### Confusion Matrix

Measure the performance of the neural network you trained and
visualize a confusion matrix.

- Did the neural network perform well on the test set?
- Did you expect this from the training loss you saw?
- What could we do to improve the performance?



#### Solutions
- does not perform well
- did we expect this from the training loss? 
- what to do to improve the prediction?
    -  add categorical variables in the model?
    -  different loss functions
    -  train for longer -- perhaps not useful
    -  make model deeper: more hidden layers



### Exercise: Explore the dataset

Let's get a quick idea of the dataset.

* How many data points do we have?
* How many features does the data have (don't count month and date as a feature)?
* What are the different types of measurements (humidity etc.) in the data and how many are there?
* (Optional) Plot the amount of sunshine hours in Basel over the course of a year. Are there any interesting properties that you notice?

date).

3. (Optional) How would we change the model if we would like to output a prediction of the precipitation in Basel in *addition* to the sunshine hours?




#### Solutions


1. How many data points do we have?

```python
data.shape
```

89 features

2. How many features does the data have (don't count month and date as a feature)?

```python
import string 
print({x.lstrip(string.ascii_uppercase + "_") for x in data.columns if x not in ["MONTH", "DATE"]})

```

3. What are the different types of measurements (humidity etc.) in the data and how many are there?

```python
import re
feature_names = set()
for col in data.columns:
    feature_names.update(re.findall('[^A-Z]{2,}', col))

feature_names
```


4. (Optional) Plot the amount of sunshine hours in Basel over the course of a year. Are there any interesting properties that you notice?

date).

```python
data.iloc[:365]['BASEL_sunshine'].plot(xlabel="Day",ylabel="Basel sunchine hours")
```

- seasonality over the months
- lots of within-day and between-day variation


5. (Optional) How would we change the model if we would like to output a prediction of the precipitation in Basel in *addition* to the sunshine hours?






## 🧠 Collaborative Notes

### What is Deep Learning?

- ChatGPT etc: built with deep learning
    - and a lot of resources (compute, engineers, data)
    - often it is not necessary to use the same amount of resources for using deep learning for research tasks
- AI landscape 
- ![](https://codimd.carpentries.org/uploads/upload_5bab7b888bb3731cba802120764c3d26.png)
- What is a neural network?
- ![](https://codimd.carpentries.org/uploads/upload_f8379a838f5549bf7cf0ade427953a4d.png)

- Ingredients for a neural network
    - input: $x_1$, $x_2$, ...: characteristics of observations. 
        - example: person's income level, has a car, ... 
    - associated weights $w_1$, $w_2$: multiplied with the inputs
        - ie, $\Sigma = x_1 * w_1 + x_2 * w_2$
    - we then want to use $\Sigma$ to predict the outcome $y$
        - the outcome is binary, ie "vote for a certain party" ($y=1$) or "do not vote for a certain party" ($y=0$)
        - for instance, we use a sigmoid activation function
        - the higher $\Sigma$, the more likely that $y=1$
- Can we have constraints on the weights?
    - (yes, with regularization?)
- In the above example, $\Sigma$ was a hidden unit, as in the figure below:
- ![](https://codimd.carpentries.org/uploads/upload_1012a124d74a45ebe8de4090d11b19cb.png)
    - in practice, there can be many hidden units, and many layers

- analogy: a neural network is a team of detectives. each neuron is a detective with a certain specialization. together, the team of detectives need to figure out whether person X is guilty.

#### How do we train a neural network?
- loss function: the difference between the prediction we make and the true value of the outcome $y$
- there are different ways to define the loss, for instance the mean squared error
    - mean squared error is for continuous outcomes

#### Why deep neural networks? -- Intuition

- why do the networks need to be deep?
    - the more complex and the more non-linear the relations between the inputs and the outputs, the more important it is to have a deep network 
- but
    - sometimes we want to understand what is going on; it is usually hard to know why a neural network comes to a certain prediction
    - we can also impose that the learned weights of the network satisfy some physical constraints. Ie, physics-inspired deep learning

An example of a deep networ: resnet50
![](https://codimd.carpentries.org/uploads/upload_fb736bc354e2eb841415d06b4a86e3e8.png)
from https://kaggle.com/kera/sresnet50

#### What is DL good for?
- pattern/object recognition
- segmenting images
- language translation/translating between one set of data and another 
- to generate synthetic data 


Examples of using DL in research
- microscopic images  -- segmentation problem
- anomaly detection

#### What DL cannot solve
- small amount of training data 
- when we need to explain how we got to the answer
- when things need to be classified that are nothing like the training data 
    - but what about transfer learning?
        - the network needs to be trained on a similar task
        - instead of transfer learning, we can also use fine-tuning to adjust the weights to our task


#### Deep learning frameworks
- pytorch
- tensorflow
- keras
    - high-level abstraction built on tensorflow


### Setting up the computer

check versions of the libraries we use

```python
from tensorflow import keras 
print(keras.__version__)
```



```python
import seaborn
print(seaborn.__version__)
```


```python
import sklearn
print(sklearn.__version__)
```

### Classification by a neural network using keras
- what is a neural network?
- how do I compose a neural network using keras?
- how do I train a network?
- how do I get insight in the learning process?
- how do I measure performance?




#### Formulate the problem

![](https://codimd.carpentries.org/uploads/upload_3de612ae4b6cf783ac628486f6311ca8.png)

we want to tell the 3 species of penguins apart.
thus, the goal is to predict the penguin species using the attributes available in the data set. 



#### Identify inputs and outputs 

```python
import seaborn as sns 
penguins = sns.load_dataset("penguins")
```


```python
penguins.head()
```


#### A bit of data exploration

```python
sns.pairplot(penguins, hue="species")
```

#### Prepare the data

handle string type easier to handle
```python
penguins["species"] = penguins["species"].astype("category")
```

```python
penguins["species"]
```


```python
# drop 2 columns; drop all rows containin NaN values
penguins_filtered = penguins.drop(columns=["island", "sex"]).dropna()
```


```python
penguins_features = penguins_filtered.drop(columns=["species"])

```

#### One-hot encoding

```python
import pandas as pd
target = pd.get_dummies(penguins_filtered["species"])
target.head()
```

```python
target.tail()
```


#### Split data in training and test set

```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(
    penguins_features, 
    target, 
    test_size=0.2, 
    random_state=0, 
    shuffle=True,
    stratify=target # make sure training and test set has similar proportions of each species
)
```


#### Build an architecture from scratch or choose a pretrianed model

```python
from tensorflow import keras 
```

```python
# random seeds
from numpy.random import seed 
seed(1)

from tensorflow.random import set_seed
set_seed(2)
```


```python
inputs = keras.Input(shape=X_train.shape[1])
```


```python
inputs
```



```python
hidden_layer = keras.layers.Dense( # Dense = all neurons are connected
    10, # number of neurons in layer
    activation="relu"
)(inputs) # syntax: keras.layers.Dense()(input): pass input to keras.layers.Dense()

```


```python
output_layer = keras.layers.Dense(
    3,
    activation="softmax"
)(hidden_layer)
```

*Softmax activation function*
- sigmoid activation function: output is between 0 and 1
- thus, in each of the output layers, the value is between 0 and 1
- but we want to have a number that *sums* up to one for each unit, which we can interpret as a probability
- we use softmax to transform the output layers into these normalized "probabilities"


```python
model = keras.Model(inputs=inputs, outputs=output_layer)
```



```python
model.summary()
```

#### Choose a loss function and optimiser

- cross-entropy for classification problem

Look at the documentation for the function `model.compile`
```python
?model.compile
```


```python
??model.compile # see the implementation of the function
```

```python
model.compile(optimizer="adam", loss=keras.losses.CategoricalCrossentropy())
```

#### Train model

```python
history = model.fit(X_train, y_train, epochs=100)
```

1 epoch = 1 full sweep through the data: If we use 100 epochs, the model sees each data point 100 times.

what's going on? -- visualize the training progress

```python
sns.lineplot(x=history.epoch, y=history.history["loss"])
```


#### Perform prediction / classification

```python
?model.predict()
```


```python
y_pred = model.predict(X_test)
```


```python
y_pred # np array?
type(y_pred)
```


```python
prediction = pd.DataFrame(y_pred, columns=target.columns)
```

This is not very handy because it is hard to get an overview of the prediction performance across all units.
One way to represent this is a confusion matrix.

Transform the prediction to the true species: extract the row index of the highest value in the prediction.
```python
predicted_species = prediction.idxmax(axis="columns")
predicted_species
```


#### Measuring performance

```python
from sklearn.metrics import confusion_matrix
```

We can measure the performance on the test data set

```python
y_test # recall that the categories are binary here
```

First, we need to know what are the true species -- again use `idxmax`.

```python
true_species = y_test.idxmax(axis="columns")

```

Now we can make the confusion matrix.

```python
matrix = confusion_matrix(true_species, predicted_species)
print(matrix)
```

We need a bit more work to make this confusion matrix easy to read.


```python
colnames = y_test.columns.values
confusion_df = pd.DataFrame(matrix, index=colnames, columns=colnames)

```

```python
confusion_df
```

We still need to label the predicted and true labels.

```python
confusion_df.index.name = "True label"
confusion_df.columns.name = "Predicted label"
```


```python
confusion_df
```

- is it good or bad? -- depends on the species
- Chinstrap not doing great: all chinstrap penguins are classified as Adelie

We can also visualize with a heatmap
```python
sns.heatmap(confusion_df, annot=True)
```


#### Tune hyperparameters
-- tomorrow

#### Share model

```python
model.save("my_first_model")
```

how to load it?

```python
pretrained_model = keras.models.load_model("my_first_model")
```

Now we can use the model to predict the categories of some new data. We use `X_test`, but we can use any "new" dataset with the same structure.

```python
y_pretrained_pred = pretrained_model.predict(X_test)
```


```python
pretrained_prediction = pd.DataFrame(y_pretrained_pred, columns=target.columns.values)
```


```python
pretrained_prediction 

```

```python
pretrained_predicted_species = pretrained_prediction.idxmax(axis="columns")
pretrained_predicted_species
```


### Monitor the training process

#### 1. Formulate / Outline the problem: weather prediction 
 
![](https://carpentries-incubator.github.io/deep-learning-intro/fig/03_weather_prediction_dataset_map.png)

Goal: predict the number of sunshine hours for tomorrow in Basel based on the weather of today. 

#### 2. Identify inputs and outputs

dataset: https://doi.org/10.5281/zenodo.5071376 -> weather_prediction_dataset_light.csv. link to file: https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1

```python
import pandas as pd 

```

```python
filename = "weather_prediction_dataset_light.csv"
data = pd.read_csv(filename)
data.head()
```

plot sunshine hours

```python
data.iloc[:365]['BASEL_sunshine'].plot(xlabel="Day",ylabel="Basel sunshine hours")

```

```python

```

```python

```

```python

```

```python

```



## Feedback


### Morning
Write down one thing that went well and that could be improved. think about the content, the pace, the instructors, the coffee, ...

#### What went well?
The hands-on part & the discussion is really great
Good pace, enough breaks 
Lot of exercises is nice and gives time to process thoughts
The pace is nice.
Nice explanations and easy to follow
Easy to follow
Nice to start with simple practical examples
So far, so good. I can follow even with limited Python experience.
Good introduction for people with no DL background, the pace so far is ok
Good pace
Nice to prectice hands on, good to have breaks to move around a bit,
Good introduction and pace, also nice have the handson part. 
Good intro and simple examples
Very interactive
Good hands on
Nice to have this amount of breaks! :)
Good instructions!
Interactive, good pacing
Good balance between doing exercise and info
Good intro with info over deap learning

#### What to be improved?
Pace can be just a little bit slower, especially when scrolling notebook away from code!
Pace could be picked up a little
Would appreciate if optional questions are also answered
Optional questions answered at least in the collab doc
Pace could be a bit faster sometimes
Maybe go back to the eventual goal a bit more during the exercise
Pace could go up a bit
Brief answer to optional questions
More learning resources would be useful for example many topics were discusses such as active learning, explainable AI etc. 
Discuss optional questions
(Briefly) also discuss optional questions / give pointers
Pace is good. An overview of some theoretical concepts would be nice. Even if it was a brief overview.
Provide information on where there is more in-depth information/reading on the topics discussed, for example why choose between one-hot or ordinal classification.
I feel a bit cold in here, but that's probably just me.
I also feel a little cold.
So do I
A little more detail on some things (e.g. why is ReLU beneficial compared to sigmoid, tanh functions etc)
Optional questions can be addressed, they can provide a better understanding on the subject.
A bit more information on deep learning intuition
Pace when using notebook couild be a bit sllower.. It is  bit colld in here.

I feel we're going a little slow. I think the requirements for the course justify a higher pace.
Have trouble keeping up with the typing on my laptop keyboard (I normally have large keyboard and two screens)
A bit higher pace or more details on for example cross entropy.
Optional questions could be clarified in short

### Sven's summary:
* It is cold
* Go faster and slower ??
* Answers to optional questions
* More resources


### Afternoon
Write down one thing that went well and that could be improved. think about the content, the pace, the instructors, the coffee, ...

#### What went well?
- Very clear how to set up a deep learning model and how Keras works
- Really liked this. Already had a mini-breakthrough in my own research thanks to this workshop. It helped that we worked through all the different parts of deep learning. 
- Nice, easy start.
- Easy to follow, nice to refer to the reference manual.
- Clear on how to set up. Keras tutorial was fun, and easy to follow.
- Easy to follow bcs of the workflow
- very clear
- Nice Lunch
- They make me feel deep learning is not magic, and that I can also build models myself. So that's a great achievement from this course! Pace is fine.
- Clear explanations and nice to have it explained by doing. Resources are great, thanks! 
- Explanations are great, thanks for both instructor
- Easy to follow and lots of resources provided
- Good variable explanations for every newly introduced function
- Great to get explanations and be able to ask questions and get answers
Clear to follow the tutorial explanation


#### What can be improved?
- Pace could be faster, less time for exercises
- More details could be given / exploring the options and play yourself could be more promoted
- Little bit slower in the afternoon compared to morning
- Pace could still be a bit faster
- In the afternoon I would have appreciated some more exercises/ time for exercises (to practice oneself, rather than doing everything together). I felt the pace in the afternoon was a bit too slow.
- maybe not every single new function needs a full explanation
- I'd like to ask some questions about my specific project. Is there a moment for that tomorrow?
- A slide with a neuron in a neural network would probably have been better than the whiteboard explanation.
- Maybe discuss exercises in pairs/groups?
- Discussing exercises in groups is a good one. Might pay off, I think.
- Good temperature in the room
- Few topics were covered. It might be nice to go over the code faster and focus more on the applications and advancing to different model specifications rather than the code itself.
- I started to get a little bit zoomed out during the afternoon. I think this was because we did take quite a lot of time for things that were pretty basic. But I was also getting tired, so it wasn't too bad. 
- More juice is always good :). Also, (for the next workshop), just make sure by the end of the first day that everyone can load their weather dataset. There is no time to explore it, and it was a bit confusing on what we expected to do in the limited time.  
could improve a  bit on speed

- Sometimes we can't see the code written if we are not up to speed. The screen is already scrolled to the last point made. 
- Some shortchuts about Juypter Notebook could be shown

### Sven's summary
* Someone already had a mini-breakthrough in their own research, yay!
* We will try to focus less on the specific syntax and more on applications
* Exercises will become longer and more complicated, you have to do and think more yourself.
*  If you are lost in the code, raise your hand, or have a look at the collaborative document
* Time for questions on your projects: during longer exercises today and  we will have 15 minutes at the end of the workshop for recap and questions.
* Emails Tessa & Machteld resolved?

## 📚 Resources
- M1 Macbook Keras install https://www.youtube.com/watch?v=o4-bI_iZKPA
- newsletter: https://esciencecenter.us8.list-manage.com/subscribe?u=a0a563ca342f1949246a9f92f&id=31bfc2303d
- poe.com: chat gpt, Claude, ...
- https://keras.io/api/losses/probabilistic_losses/
- [tensorflow playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.48778&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
- google colab https://colab.research.google.com/
- miniconda for m1 (choose Miniconda3 macOS Apple M1 64-bit): https://docs.conda.io/en/latest/miniconda.html
- Play around with explainable AI methods: https://github.com/dianna-ai/dianna
- Learn more about active learning: https://towardsdatascience.com/active-learning-5b9d0955292d
- [The community of Research Software Engineers from Dutch universities](https://nl-rse.org/)
- [Softmax function](https://deepai.org/machine-learning-glossary-and-terms/softmax-layer)
- [Modelzoo:Discover open source deep learning code and pretrained models](https://modelzoo.co/)
- [keras API documentation](https://keras.io/api/)
    - [`fit` method](https://keras.io/api/models/model_training_apis/#fit-method)
- [Intuition for Adam Optimizer](https://www.geeksforgeeks.org/intuition-of-adam-optimizer/)
   - Maybe first read about [AdaGrad](https://www.geeksforgeeks.org/intuition-behind-adagrad-optimizer/)
 - dataset for monitoring training process: https://doi.org/10.5281/zenodo.5071376
