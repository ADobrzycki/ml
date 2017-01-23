import numpy as np

from regression import LinearRegression
from sklearn import datasets
from misc import vector
from math_functions import mean_squared_error

def test_predict():
	dataset = datasets.load_diabetes()
	dataset, target = dataset['data'], dataset['target']
	data = vector(dataset[0])

	lr = LinearRegression()
	print(lr.predict(data))

def test_mean_squared_error():
	target = np.array([1,2,3])
	predictions = np.array([2,3,4])
	print(mean_squared_error(target, predictions))

def test_get_weights():
	dataset = datasets.load_diabetes()
	dataset, target = dataset['data'], dataset['target']

	lr = LinearRegression()
	weights = lr.get_weights(dataset, target)
	print(weights)

	data = vector(dataset[0])
	prediction = weights.transpose().dot(data)
	print(target[0])
	print(prediction)

	data = vector(dataset[1])
	prediction = weights.transpose().dot(data)
	print(target[1])
	print(prediction)



if __name__ == '__main__':
	#test_predict()
	#test_mean_squared_error()
	test_get_weights()