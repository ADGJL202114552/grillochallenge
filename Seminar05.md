# Seminar 5: Convolutional Neural Networks (CNN)

The goal of this seminar is twofold:

* demonstrate how to create a CNN model to perform multi-class classification, and
* explore optimization strategies, such as dropout, model checkpoints, and deep architectures to improve classification accuracy.

We first consider the task of image classification for handwritten digit recognition, based on the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset. We then explore the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) (for colour images) and the Arturo Lozano's radiology and medical image dataset based on [MedMNIST](https://medmnist.com/) datasets.

## Activity 1: MNIST classification

MNIST stands for the *Modified National Institute of Standards and Technology* and is a [dataset](http://yann.lecun.com/exdb/mnist/) of 60,000 small square 28×28 pixel images of handwritten single digits between 0 and 9.

<img src="./figs/mnist_plot.png" width=400>

Here is an example of how **black and white images** (obtained from MNIST dataset) are represented:

<img src="./figs/mnist3.png" width=300>

This image has a resolution of 28x28 and 1 channel (or `(28,28,1)`). Each pixel is a number between 0 and 1, where 0 is shown as black, 1 is white and any number inbetween is gray.  The image data is presented as a flattened long vector with a length of 784 (28x28).  

**Action:**

* Open the [W05_CNN_MNIST](https://colab.research.google.com/drive/1iO_pZEArmfkm1dJQ4grWQOl3akWRlTQ_#offline=true&sandboxMode=true) notebook for a walkthrough CNN implementation. We start with a baseline model (Dense layer) and progressively add more layers and other optimization strategies to improve the model. 

## Activity 2: CIFAR-10 classification

In this activity, we will explore the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). It consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

<img src="./figs/cifar-10.png" width=400>

Differently for black and white images, in a **colour image**, each pixel has 3 dimensions (i.e. the RGB channels). The shape of this image is 178x218x3. Each element in one pixel is a number between 0 and 255.  

<img src="./figs/cat.png" width=400>

**Action:**

* Open the [W05_CNN_CIFAR](https://colab.research.google.com/drive/1Y2cPEBxMtNcy6cLtahX0tQNMtnvcremV#offline=true&sandboxMode=true) notebook for a walkthrough CNN implementation. Again, we will try to improve our CNN model by exploring different combination of layers and optimization parameters.

## Homework:

* Both notebooks have activities left as *self-study*. They include expading the models to include more layers and experimenting with different techniques for performance improvement. Whether a given approach is suitable to your model depends on the data you are using and your choices regarding the model's architecture. 

* Have a look at the first two references below for i) a comprehensive recap of all the concepts discussed this week, and ii) an animation of a CNN processing MNIST data, with emphasis on the patterns learned by each layer over time.

* Have a look at the [W05_MedMNIST.ipynb](https://colab.research.google.com/drive/1iHj4awEOOR2rDucXE7Q8Ez47-RawO5zu#offline=true&sandboxMode=true) notebook showing a CNN implementation for a modified version of the [MedMNIST](https://medmnist.com/) dataset. This code explores further resources (callbacks) for improving model's performance, such as **Early stopping** (to stop training when a monitored metric has stopped improving) and **Reduce LR On Plateau** (to reduce learning rate when a metric has stopped improving), besides the **Model Checkpoint** we have used in the MNIST example.

<img src="./figs/MedMNIST.png" width=300>

## References

* [TensorFlow, Keras and deep learning, without a PhD](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0).
* [ConvNetJS MNIST demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html).
* [TensorFlow 2 tutorial: get started in Deep Learning with tf.keras](https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/).
* [How to develop a CNN for MNIST handwritten digit classification](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/).
* [Guide to MNIST datasets for fashion and medical applications](https://analyticsindiamag.com/fashion-and-medical-mnist/)

