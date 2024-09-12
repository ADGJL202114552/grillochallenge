# Introduction to TensorFlow

We will be using [Google Colab](https://colab.research.google.com/) in our classes, so **you don't need to install any software into your machine**. 

But you if want to do it, please follow the links in the *Installation* section.

**Contents**:
* [What is TensorFlow?](#what-is-tensorflow)
* [TensorFlow architecture](#tensorflow-architecture)
* [What is a tensor?](#what-is-a-tensor?)
* [Installation](#installation)

## What is TensorFlow?

[TensorFlow](https://www.tensorflow.org/) is an open source library for machine learning research. Its core elements are written in C++ language and are able to execute on CPUs, GPUs and [TPUs](https://cloud.google.com/tpu/docs/tpus) (Tensor Processing Units). It provides a Python API (among other APIs) for full access to these core elements.

Just like other libraries, such as Numpy and Scipy, TensorFlow is designed to make the implementation of machine learning programs easier and faster.

## TensorFlow architecture

The TensorFlow 2.0 architecture is organised in two major blocks: *Training* and *Deployment*.

<!--- ![tf](https://github.com/lse-st449/lectures2020/raw/master/Week01/Class/graphs/tf_architecture.png) -->

<img src="./figs/tf2_architecture.png" width=600>

*Training* comprises elements for data ingestion, model building and validation (through Keras and Estimators) and distributed training over different hardware accelerators (CPUs, GPUs and TPUs). Models can be saved and exported for sharing purposes (through [TF Hub](https://tfhub.dev/)). The *Deployment* of saved models can be done on a variety of platforms and other languages.

All **TensorFlow modules and classes** can be found [here](https://www.tensorflow.org/api_docs/python/tf).

You may find some tutorials and book examples based on older versions of TensorFlow (e.g. `1.x`). [This blog](https://blog.tensorflow.org/2019/01/whats-coming-in-tensorflow-2-0.html) provides an overview of these elements and discusses some differences between TensorFlow 1.x and TensorFlow 2.x.

#### Low level APIs and high level APIs

You can either code in the low-level TensorFlow API (**TensorFlow Core**) or in the high-level APIs.

Although the high-level APIs provide a much simpler and consistent interface, it is still beneficial to learn how to use Tensorflow Core for the following reasons, as mentioned in the official documents:

* Experimentation and debugging are both more straightforward when you can use low level TensorFlow operations directly.
* It gives you a mental model of how things work internally when using the higher level APIs.

## What is a tensor?

A tensor is an n-dimensional array of an uniform type (called a [dType](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType)). You can think it as a Python list or a Numpy array but with different data types.

A scalar is treated as a zero-dimensional (or `rank-0`) tensor, an array is treated as a one-dimensional (`rank-1`) tensor, and a matrix is treated as a two-dimensional (`rank-2`) tensor.

<img src="./figs/tf2_tensors.png" width=600>

Refer to the [Introduction to Tensors](https://www.tensorflow.org/guide/tensor) tutorial for a detailed view of tensor types and related operations. **Notice that you can run the code on Google Colab or download it**.

## Installation

During seminar sessions, we will be using Python notebooks in [Google Colaboratory](https://colab.research.google.com), which is a cloud service for machine learning research. The reasons why we choose it as an additional tool are as following:

* TensorFlow is set up by default.
* Google Colab provides a free GPU/TPU and allows for easy installation of additional libraries.

**You don't need to install any software for attending the seminar sessions**. Just make sure you have access to Google Colab and some cloud storage (e.g. Google Drive, OneDrive etc).

However, if you want to set up everything on your local machine, please follow these instructions:

### Notebooks vs integrated development environments

Notebooks (Jupyter, iPython, etc) are suitable for presenting results or for smaller projects. If you implement something of a magnitude beyond your coursework, you may write your code in an integrated development environment (IDE) (e.g. PyCharm and Spyder) which has better debugging tools and makes coding much more enjoyable. In this course, you can use either.

If you want to set up everything on your local machine, please follow these links:

* Install Anaconda [here](https://docs.anaconda.com/anaconda/install/)
* Install PyCharm [here](https://www.jetbrains.com/help/pycharm/installation-guide.html) (Community version is fine).
* Check [this link](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html) for configuring a Conda environment in PyCharm.
* Install Spyder [here](https://docs.spyder-ide.org/current/installation.html). You can also play with a Web-based version (no installation needed).

### Hardware requirements for the GPU version

There are a CPU version and a GPU version of TensorFlow. The GPU version only supports NVIDIA® GPU card with CUDA® Compute Capability 3.5 or higher.  Before installing, you will need to check your GPU card. Generally speaking, if you are installing on your laptop, we recommend you to install the CPU version. If you've got one of [these GPU cards](https://developer.nvidia.com/cuda-gpus), feel free to install the GPU version.

### Installation guides

* [Install TensorFlow 2](https://www.tensorflow.org/install)
* [Install TensorFlow with pip](https://www.tensorflow.org/install/pip)
* Refer to [this guide](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/) for installing TensorFlow in Anaconda.
* There is a [guide here](https://www.tensorflow.org/extras/cert/Setting_Up_TF_Developer_Certificate_Exam.pdf) for setting up TensorFlow in PyCharm.
* There is a tutorial [here](https://medium.com/@pushkarmandot/installing-tensorflow-theano-and-keras-in-spyder-84de7eb0f0df) and [here](https://newbedev.com/how-to-use-tensorflow-on-spyder) on how to use TensorFlow on Spyder.
