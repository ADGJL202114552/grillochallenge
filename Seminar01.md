# Seminar 1: Introduction to deep learning

In this first seminar, we will cover **linear and logistic regression models**, which can be considered as building blocks for neural networks.

## Activity 1 (tutorial): Linear regression

We intend to model a linear relationship between a dependent variable `y` and one or more independent variables (or features) `X`. The basic elements of a **linear regression model** consist of a set of features (`X`), the target variable (`y`), a set of weights (`w`) and the bias parameter (`b`), as depicted below.

<center>
<img src="./fig/w01_fig1a.png" width=200>
</center>

For **predicting** the target variable (`y`), our linear model can be considered as an *affine transformation* of the input features (`X`), characterised by a *linear transformation* of features via weighted sum, combined with a *translation* via the added bias.

`[from the D2L reference book]:`
Given features of a training dataset `X` and corresponding (known) labels `y`, the goal of linear regression is to find the weight vector `w` and the bias term `b` that given features of a new data example sampled from the same distribution as `X` , the new example's label will (in expectation) be predicted with the lowest error.

In order to find the best parameters (or model parameters) `w` and `b`, we need:

* a quality measure for some given model: *loss function*, and
* an algorithm for updating the model to improve its quality: the simplest one is *gradient descent* and its variations.

### Your action

1. Click to open the [w01_linear-regression.ipynb](https://colab.research.google.com/drive/1JoO-ADSExNjRxySiB3hAHiBb2MHJRbPz#offline=true&sandboxMode=true) notebook on Google Colab.
2. Go the the first part (approach), which is a walkthrough implementation of a linear regression model. Make sure you understand each step.
3. Then, go to the second part (approach) and observe how TensorFlow high-level APIs are used to a concise implementation of the same linear regression model.

## Activity 2 (tutorial): Logistic regression

Logistic regression allow us to solve **classification** problems: instead of asking *how much* or *how many*, we want to discover *which one*. 

Often, we can deal with two subtly different problems: (i) those where we are interested only in hard assignments of examples to categories (classes); and (ii) those where we wish to make soft assignments, i.e., to assess the probability that each category applies. The distinction tends to get blurred, in part, because often, even when we only care about hard assignments, we still use models that make soft assignments.

In order to estimate the **conditional probabilities associated with all the possible classes**, we need a model with multiple outputs, one per class. Each output will correspond to its own *affine* function. Just as in linear regression, logistic regression is also a single-layer neural network: the calculation of each output depends on all inputs, so the output layer can be also described as a fully-connected layer. 

The main difference is that, for logistic regression, we use an **activation function** to map the inputs into the number of classes we have. For instance, in a **binary classification problem**, we can apply a **sigmoid activation** to allow our prediction function to output values between 0 and 1. For a **multiclass classification problem**, we can use other activation functions, such as **softmax** (see the Homework section).

The most common **loss function** used for classification problems is **cross-entropy**: it is the expected value of the loss for a distribution over labels. TensorFlow implements different loss functions in the [tf.keras.losses](https://www.tensorflow.org/api_docs/python/tf/keras/losses) module.

## Your action

1. Click to open the [w01_logistic-regression.ipynb](https://colab.research.google.com/drive/1pGSKv9LAqXhOosWzfACjRRISz6ecE3Pb#offline=true&sandboxMode=true) notebook.
2. Download the modified version of the `titanic` dataset (from the `data` folder) and upload it into your Colab notebook.
3. Go the the first part, which is a walkthrough implementation of a logistic regression model. Make sure you understand each step.
4. Have a look at the second part, where we rely on `Keras` for implementing the same model.

## Homework (self-study)

A) Hyperparameter search

1. What would happen if we were to initialize the weights to zero? Would the algorithms still work? Change the linear or logistic regression code to reflect that.
2. Try adjusting the hyperparameters, such as the batch size, number of epochs, and learning rate, to see what the results are.
3. Review the TensorFlow documentation to see [what loss functions](https://www.tensorflow.org/api_docs/python/tf/keras/losses) and [initialization methods](https://www.tensorflow.org/api_docs/python/tf/keras/initializers) are provided. Play with other loss functions.
4. Check out the [TensorFlow `data` API](https://www.tensorflow.org/guide/data) documentation for other available datasets.

B) Softmax multiclass classification

Click to open the [w01_softmax-regression.ipynb](https://colab.research.google.com/drive/1J3amDshpV-19GvArP7HDemy9uZGdxOEr#offline=true&sandboxMode=true) notebook for a walkthrough example of a multiclass classification problem using the *Fashion MNIST* dataset and the *softmax* activation function.

## References

* Brilliant: [Linear regression](https://brilliant.org/wiki/linear-regression/)
* TensorFlow tutorial: [Basic regression: Predict fuel efficiency](https://www.tensorflow.org/tutorials/keras/regression)
* Machine Learning: [How to implement Linear Regression in TensorFlow](https://www.machinelearningplus.com/deep-learning/linear-regression-tensorflow/)
* Kaggle: [Logistic Regression with TensorFlow](https://www.kaggle.com/autuanliuyc/logistic-regression-with-tensorflow)
* Machine Learning Mastery: [A Gentle Introduction to Mini-Batch Gradient Descent and How to Configure Batch Size](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)
* Towards Data Science: [Batch, Mini Batch & Stochastic Gradient Descent](https://towardsdatascience.com/batch-mini-batch-stochastic-gradient-descent-7a62ecba642a)
