
__author__ = "Madeleine Sheehan"


from decisiontree import DecisionTree
import numpy as np

class RandomForest:
	def __init__(self, iterations, m, min_leaf=1, max_depth="inf"):
		'''
		iterations - number of random trees to use
		m - number of random features to use at each split for each decision tree
		min_leaf and max_depth are optional decision tree parameters
		'''
		self.ITERATIONS = iterations
		self.MIN_LEAF = min_leaf
		self.M = m
		self.MAX_DEPTH = max_depth

	def train(self, data, labels):
		self.trees = []
		for i in range(self.ITERATIONS):
			inds = np.random.choice(np.arange(len(data)), len(data))
			self.trees.append(DecisionTree(min_leaf=self.MIN_LEAF, m=self.M, max_depth=self.MAX_DEPTH))
			self.trees[-1].train(data[inds],labels[inds])
	
	def predict(self, data):
		# Note only works for binary classification right now. TODO change return to find most common so it
		# it can generalize to other classification problems
		predictions = []
		for tree in self.trees:
			predictions.append(tree.predict(data))
		return np.round(np.sum(predictions,0)/len(predictions))