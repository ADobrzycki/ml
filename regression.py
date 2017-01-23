import numpy as np

from misc import vector

class LinearRegression:
	def __init__(self):
		self.weights = None

	def predict(self, data):
		self.weights = vector(np.ones(data.shape[0]))
		prediction = self.weights.transpose().dot(data)
		return prediction