import numpy as np

def mean_squared_error(target, predictions):
	diff_dist_sum = 0	
	for prediction in predictions:
		diff_dist_sum +=  np.linalg.norm(prediction - target, 2)
	constant = 1 / len(predictions)
	return np.multiply(constant, diff_dist_sum)