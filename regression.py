import numpy as np

from numpy.linalg import inv
from misc import vector

class LinearRegression:
	def __init__(self):
		self.weights = None

	def predict(self, data):
		self.weights = vector(np.ones(data.shape[0]))
		prediction = self.weights.transpose().dot(data)
		return prediction

	def get_weights(self, dataset, target):
		# uses solution to mean squared error equation optimisation (5.12)
		return inv(dataset.transpose().dot(dataset)).dot(dataset.transpose()).dot(target)