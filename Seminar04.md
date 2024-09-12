# Seminar 4: Advanced optimization algorithms

* Folder link: https://drive.google.com/drive/folders/12uEqTmab1seQIZmDq8M7MHP8-ynT5Aua?usp=sharing

## Homework (Week03): Higher-dimensional and noisy data

* Open the `W3HWsol.ipynb` notebook on Google Colab.

### Previous Activities 

1. Repeat the process to non-linear NNs:
	* Use `Model1_make` to construct the NN we constructed earlier in the class
	* `Model_fit` contains batch size as an input
2. Train `m1_trained_5` with 500 more epochs. See what happens. Key word: stability.
3. Add another RELU at the end with `kernel_initializer=initializers.Ones()` and repeat the experiment with different batch sizes we have done previously, observe the behaviour:
	* It probably won't look nice (note that you need to set seed, which has been covered in `initialisation_tf()`, so that the results are replicable)
	* Think about what has been covered in the class on initialisers and try again.
4. Run `Model_fit` with 200 batches for 800 epochs, and with 5 batches for 3000 epochs. Look at the result and conjecture the convergence speed.

### Main Takeaways

* Batch size:
	* When batch size is small, there are many batches to compute, so takes more time. But because we compute many batches, accuracy and stability improve. 
	* The stability is important, especially as we scale up the model dimensions (from linear to 1LNN, then 2LNN).
	* The case of larger batch size suffers instability: when stability is guaranteed, it may still converge, at a slower rate in terms of epoch --- but more or less similar in terms of overall computing time, as the computing time is reduced due to the smaller number of batches.

* Initialisation:
	* Randomness occurs and it showcases stability.

<hr>

## Activity 1: Learning rate and momentum

* Open the [W4MainNB1_students_version.ipynb](https://colab.research.google.com/drive/1Kyi4MZef6xmqjQWNm_-IDrHaCJ9WjHNi#offline=true&sandboxMode=true) notebook on Google Colab.
* We will look closely at the impact of `learning_rate`, and later on momentum to the same training task.
* We code-up some basic functions to run these experiments.

### Activities 

1. Run linear regression models using `linear_regression_make` with `learning_rate=0.1, 0.05, 0.01` and for `epochs=100`. Observe the loss trajectory.
2. Now run the same models using `momentum=0.9`.
3. Final remarks: we can look at the similarities amongst the predictions by the `MakePrediction` class.

## Activity 2: Additional gradient methods

* Open the [W4MainNB2_students_version.ipynb](https://colab.research.google.com/drive/1lXmA0VmStBH9CPbaXwaJD98ie7oFRHTI#offline=true&sandboxMode=true) notebook on Google Colab.
* We will look at `Adagrad`, `RMSprop`, and `Adam`.
* We will look closely to the weights and appreciate what the algorithm does at a layer-weights level.

### Activities 

1. Complete the `m3` and `m4` with `RMSprop` and `Adam`.
2. Observe the loss trajectory and predictions.
3. Observe the weights within each layer.

## Activity 3: A taste of dropout

* Open the [W4MainNB3_students_version.ipynb](https://colab.research.google.com/drive/1ibHRCi5QuNPcV7yscY0sjlmbzDTgk7W0#offline=true&sandboxMode=true) notebook on Google Colab. 
* We will look at the dropout method at an introductory level: 
	* We reduce the sample size and have a look at the SGD performance.
	* This prepares you for further investigation in the homework.

### Activities 

1. Code two models: `m1` as SGD with dropout, and `m2` as SGD without dropout. 
2. Compare the loss trajectories.
3. Compare the model weights.

## Homework: Further exercise on dropouts

1. Open the [W4HW.ipynb](https://colab.research.google.com/drive/1hnsrB7lAfDPurfYzQPagYewj0-6TT4cx#offline=true&sandboxMode=true) notebbok on Google Colab.
2. Train 4 models specified by `Model2_make` using `Adam`: one should be with no dropout, the other three with dropout rates 0.1, 0.2, and 0.5. respectively.
3. Merge the histories and observe the loss and validation loss.
	* Expect results to behave uncomfortably --- but at least some should do a better job than the others.
4. Have a look at the weights
	* Remember to use the tricks performed in the class.


