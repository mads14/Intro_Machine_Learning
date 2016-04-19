
__author__ = "Madeleine Sheehan"


from decisiontree import DecisionTree
import numpy as np

class RandomForest:
	def __init__(self, iterations, m, min_leaf=1):
		#m = number of subsamples
#		 self.SAMPLES = samples
		self.ITERATIONS = iterations
		self.MIN_LEAF = min_leaf
		self.M = m

	def train(self, data, labels):
		#get m random samples, first without replacement
		self.trees = []
		for i in range(self.ITERATIONS):
			inds = np.random.choice(np.arange(len(data)), len(data))
			self.trees.append(DecisionTree(self.MIN_LEAF, self.M))
			self.trees[-1].train(data[inds],labels[inds])
	
	def predict(self, data):
		predictions = []
		for tree in self.trees:
			predictions.append(tree.predict(data))
			
		return np.round(np.sum(predictions,0)/len(predictions))