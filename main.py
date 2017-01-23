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
	lr.predict(data)

def test_mean_squared_error():
	target = np.array([1,2,3])
	predictions = np.array([2,3,4])
	print(mean_squared_error(target, predictions))

if __name__ == '__main__':
	test_mean_squared_error()