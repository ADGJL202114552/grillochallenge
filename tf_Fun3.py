import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

from sklearn import model_selection as skms

import numpy as np
import pandas as pd
from pandas import DataFrame as df
import seaborn as sns
import matplotlib.pyplot as plt

def initialisation_tf(seed_value=0):
	print(tf.__version__)
	#If you want the results to be reproducible.
	seed_value=0
	tf.random.set_seed(seed_value)
	np.random.seed(seed_value)
	return 

def linear_regression_make(feature_normaliser,learning_rate=0.01,momentum=0,name='linear_regression'):
	linear_regression = keras.Sequential(name=name)
	linear_regression.add(feature_normaliser)
	linear_regression.add(layers.Dense(units=1, name='dense'))
	# linear_regression.summary()
	linear_regression.compile(
	    optimizer=tf.optimizers.SGD(learning_rate=learning_rate,momentum=momentum),
	    loss='mean_squared_error')
	return linear_regression

def Model1_make(feature_normaliser,learning_rate=0.01,momentum=0,name='Two_layer_model'):
	model = keras.Sequential(name=name)
	model.add(feature_normaliser)
	model.add(layers.Dense(8,activation='sigmoid',name='sigmoid1'))
	model.add(layers.Dense(4,activation='sigmoid',name='sigmoid2'))
	model.add(layers.Dense(1,name='dense_after_sigmoid'))
	model.compile(
    optimizer=tf.optimizers.SGD(learning_rate=learning_rate,momentum=momentum),
    loss='mean_squared_error')
	return model

def Model_fit(model,X_train,y_train,X_test=None,y_test=None,batchnumber=50,epochs=50,verbose=0):
	if X_test is not None:
		history=model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=int(X_train.shape[0]/batchnumber),
    verbose=verbose,
    validation_data = (X_test,
                      y_test))
	else:
		history=model.fit(
    	X_train,
    	y_train,
    	epochs=epochs,
    	batch_size=int(X_train.shape[0]/batchnumber),
    	verbose=verbose)
	return history,model

class MakePrediction:
	def __init__(self,models,X_actual,y_actual):
		self.models 	= models
		self.y_actual = y_actual
		self.X 				= X_actual
	def predict(self,additional_input=None):
		self.pred=[]
		if additional_input is None:
			for m in self.models:
				self.pred.append(df(m.predict(self.X)))
		else:
			for m in self.models:
				self.pred.append(df(m.predict(additional_input)))		
	def run_basic_plot(self,names,color=None):
		self.predict()
		Dy=df(self.y_actual).reset_index(drop=True).sort_values('sig')
		D=(pd.concat([Dy]+self.pred,axis=1)).reset_index(drop=True)
		D.columns=['Actual']+names
		if color is None:
			fig=D.plot(grid=True)
		else:
			fig=D.plot(grid=True,color=color)
		return fig

class TrainAndCollect:
	def __init__(self,
		model,
		X_train,
		y_train,
		epochs=200,
		verbose=0):
		self.model = model
		self.X_train = X_train
		self.y_train = y_train
		self.verbose = verbose
		self.epochs = epochs
		self.length_of_weights = len(model.weights)-3
		self.collection = [df()]*self.length_of_weights

	def train_memo(self):
		for epo in range(self.epochs):
			self.model.fit(
  		  self.X_train,
    		self.y_train,
    		epochs=1,
	    	verbose=self.verbose)
			for entry in range(self.length_of_weights):
				self.collection[entry] = pd.concat([self.collection[entry],
					df({epo:self.model.weights[entry+3].numpy().flatten()})],axis=1)

		return self.collection

	def plot_one(self,entry):
		if self.collection[entry].shape[0]>20:
			print('too many lines!')
		else:
			return self.collection[entry].transpose().plot()


class ImplementTrainAndCollect:
	def __init__(self,
		X_train,
		y_train,
		models):
		self.X_train = X_train
		self.y_train = y_train
		self.models = models
		self.collection = dict()
		self.basecolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(self.models)]

	def implement_all(self):
		for m in self.models:
			TAC=TrainAndCollect(m,self.X_train,self.y_train)
			TAC.train_memo()
			self.collection.update({TAC.model.name:
				TAC.collection})

		return None

	def plot_one(self,entry,figsize=(10,5),grid=True):
		def make_color(n_repeat=self.collection[self.models[0].name][entry].shape[0]):
			a=[]
			for b in self.basecolor:
				a+=([b]*n_repeat)
			return a

		a=[]		
		for m in self.models:
			name=m.name
			D1=self.collection[name][entry].transpose()
			a.append(D1.add_suffix('_'+name))

		D0=pd.concat(a,axis=1)
		self.D0 = D0
		return D0.plot(figsize=figsize,grid=grid,color=make_color())

