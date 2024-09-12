# Seminar 2: Feedforward neural networks

The materials for this seminar session cover the following topics:

* single and multilayer perceptron examples used to solve the XOR problem.
* the design and validation of multilayer perceptron (MLP) neural networks.

## Activity 1 (reference tutorial): XOR problem

This example code illustrates how:

1. a single layer perceptron can be implemented and used to solve AND, OR, and NAND functions, and
2. OR, AND, and NAND functions can be combined into a multilayer perceptron to solve the XOR problem.

### Your action

1. Click to open the [w02_XOR-Perceptron.ipynb](https://colab.research.google.com/drive/1T_o7pYe0iEukO2AURoDtZUHY6d4-lCSH#offline=true&sandboxMode=true) notebook on Google Colab.
2. Go the the implementation of the `Perceptron` class to understand its structure and use for solving AND, OR, and NAND functions.
3. Have a look at the `MLP` class and the implementation of the XOR function.

## Activity 2: multilayer perceptron (in-class activity)

In this activity, we will look at some multilayer perceptron implementations applied to different data sets (satellite data, image processing).

Open the [w02_MLP_tf_keras_activity.ipynb](https://colab.research.google.com/drive/1Cj2RHyk5-G8dQlh2xma6pQ-3HuaTkVEk#offline=true&sandboxMode=true) notebook and make a copy, so you can fill in the gaps in the code.

## Homework

A) Hyperparameter search / model exploration

1. What is the best result you can get by optimising over all the hyperparameters (learning rate, number of epochs, number of hidden layers, number of hidden units per layer) separately and then jointly?
2. What is the smartest strategy you can think of for structuring a search over multiple hyperparameters?
3. Try out different activation functions. Which one works best?
4. Try different schemes for initialising the weights. What method works best?

B) Perceptrom from scratch

<img src="./figs/w02_hw.png" width="400">

1. Open the [w02_PerceptronLearning_hw.ipynb](https://colab.research.google.com/drive/1jmxwpwrgNZm3N2p8MH22Ud2K-IoM6CEr#offline=true&sandboxMode=true) code and implement your own version of a Perceptron algorithm to solve a binary classification problem.

C) Additional MLPs examples

1. Binary classification using the IMDB movie dataset: [w02_MLP-IMDB.ipynb](https://colab.research.google.com/drive/1q-mwFuHSPYU1_qKtcwUkQfoIZLjtpBg8#offline=true&sandboxMode=true)
2. Multiclass classification using the Reuters newswire dataset: [w02_MLP-Reuters.ipynb](https://colab.research.google.com/drive/172cyr3e5TPSASFRPS5PdxTwMssfT6n5c#offline=true&sandboxMode=true)

## References

* Paper: [An introduction to computing with neural nets](https://ieeexplore.ieee.org/abstract/document/1165576)
* Machine Learning Mastery: [How to Configure the Number of Layers and Nodes in a Neural Network](https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/)
