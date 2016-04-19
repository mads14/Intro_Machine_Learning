import numpy as np
import scipy.io as sio
from sklearn.cross_validation import KFold

class DecisionTree:
	def __init__(self, min_leaf=1, m=None, max_depth="inf", curr_depth=1):
		self.MIN_LEAF = min_leaf
		self.M = m
		self.CURR_DEPTH = curr_depth
		self.MAX_DEPTH = max_depth
	
	def train(self, data, labels):
		'''Grows a decision tree by constructing nodes. Using the impurity and segmenter
		methods, attempts to find a configuration of nodes that best splits the input 
		data. This function figures out the split rules that each node should have and 
		figures out when to stop growing the tree and insert a leaf node. DecisionTree 
		stores the root node of the resulting tree so you can use the tree for classifica- 
		tion later on.'''

		node_label = np.argmax(np.bincount(labels))
		self.rootNode = Node(label=node_label, size=len(data), min_leaf=self.MIN_LEAF, tree=self, m=self.M)

		#split if you haven't reached max depth:
		if self.CURR_DEPTH < self.MAX_DEPTH:
			self.rootNode.segmenter(data, labels, node_label)
		
		if self.rootNode.split_rule != None: 
			#walk down right side of tree
			right = DecisionTree(self.MIN_LEAF, m=self.M, max_depth=self.MAX_DEPTH, curr_depth=self.CURR_DEPTH+1)
			right_inds, left_inds = self.rootNode.split(data, self.rootNode.split_rule)
			right.train(data[right_inds],labels[right_inds])
			self.rootNode.right = right.rootNode
			
			#walk down left side of tree
			left = DecisionTree(self.MIN_LEAF, m=self.M, max_depth=self.MAX_DEPTH, curr_depth=self.CURR_DEPTH+1)
			left.train(data[left_inds],labels[left_inds])
			self.rootNode.left = left.rootNode
			
		else:
			return self.rootNode
	
	def predict(self, data):
		'''Given a data point, traverse the tree to find the best label to classify the 
		data point as. Start at the root node you stored and evaluate split rules at each
		node as you traverse until you reach a leaf node, then choose that leaf node's label
		as your output label.'''
		labels = np.ones(len(data))*.5
		if self.rootNode.split_rule != None:
			right_inds, left_inds = self.rootNode.split(data, self.rootNode.split_rule) 
			
			right_data = data[right_inds]
			right_labels = self.rootNode.right.tree.predict(right_data)
		
			left_data = data[left_inds]
			left_labels = self.rootNode.left.tree.predict(left_data)
			
			#assign labels
			labels[right_inds] = right_labels
			labels[left_inds] = left_labels
			return labels
		else:
			labels = [self.rootNode.label]*len(data)
			return labels


	

	def prune(self, data, labels):
		''' data = prune set data and labels = prune set labels'''
		# 
		if self.rootNode.split_rule != None:
			right_inds, left_inds = self.rootNode.split(data, self.rootNode.split_rule) 
			
			right_errors = np.sum(self.rootNode.label - labels[right_inds])
			right_labels, r_errors = self.rootNode.right.tree.prune(data[right_inds], labels[right_inds])
			
			
			left_errors = np.sum(self.rootNode.label - labels[left_inds])
			left_labels, l_errors = self.rootNode.left.tree.prune(data[left_inds], labels[left_inds])
			
			# errors is the number of errors if all classified as parent node
			errors = np.sum(np.abs(self.rootNode.label - labels))
			
			# if errors from classifying children higher than errors from not classifying children
			if r_errors+l_errors > errors:
				
				# remove split rule. Make parent node a new leaf.
				self.rootNode.split_rule=None
				return labels, errors
			
			# otherwise keep split.
			else:
				labels[right_inds] = right_labels
				labels[left_inds] = left_labels
				return labels, r_errors+l_errors
		else:
			labels_ = [self.rootNode.label]*len(data)
			
			#errors = number misclassified in leaf
			errors = np.sum(np.abs(self.rootNode.label - labels))
			return labels_, errors

class Node:
	def __init__(self, label=None, size=None, tree=None, min_leaf=1, m=None):
		'''
		- split_rule: A length 2 tuple that details what feature to split on at a node,
		as well as the threshold value at which you should split at. The former can 
		be encoded as an integer index into your data point's feature vector.
		- left: The left child of the current node.
		- right: The left child of the current node.
		- label If this field is set, the Node is a leaf node, and the field contains the 
		label with which you should classify a data point as, assuming you reached this node
		during your classification tree traversal. Typically, the label is the mode of the 
		labels of the training data points arriving at this node.'''
		
		self.label = label
		self.size = size
		self.tree = tree
		self.MIN_LEAF = min_leaf
		self.M = m
		self.split_rule = None
		self.left = None
		self.right = None



	def split(self, data, split_rule):
		'''Split the data using a split rule. The split rule is a two-element tuple 
		with split_rule[0] = the split feature, and split_rule[1] = the split criteria.
		The data is assigned to right indices if data['split_feature'] > split_criteria.
		'''

		split_feature = split_rule[0]
		split_criteria = split_rule[1]
		if type(split_criteria) == str:
			right_inds = data[:,split_feature] == split_criteria
		else:
			right_inds = data[:,split_feature]>split_criteria

		left_inds = np.invert(right_inds)								   
		return right_inds, left_inds
	
	def segmenter(self, data, labels, node_label): 
		'''A method that takes in data and labels. When called, it finds the best split 
		rule for a Node using the impurity measure and input data. There are many 
		different types of segmenters you might implement, each with a different method 
		of choosing a threshold. The usual method is exhaustively trying lots of different 
		threshold values from the data and choosing the combination of split feature and 
		threshold with the lowest impurity value. The final split rule uses the split 
		feature with the lowest impurity value and the threshold chosen by the segmenter. 
		Be careful how you implement this method! Your classifier might train very slowly 
		if you implement this badly.'''
		
		min_impurity = 1.0

		#break by returning none if we have found a leaf node:
		#if size <= min leaf return node
		if len(data)<=self.MIN_LEAF:
			return None

		# test whether leaf node
		if np.sum(labels==0) ==0 or np.sum(labels==1) == 0:
			return None
			# return self#Node(label = node_label, size=len(data))
		

		#if random forest, randomly select M features to sort on. Else use all.
		if self.M: features = np.random.choice(data.shape[1], self.M, False)
		else: features = range(data.shape[1])
		# features = range(data.shape[1])
		
		#get all possible splits, test to see which produces most info gain.
		for feature in features:	
			splits = np.sort(np.unique(data[:,feature]))
			#TODO if string should be split in splits, (unless 2 categories)
			if (type(splits[0]) == str) & len(splits)>2:
				splits = splits
			else: 
				splits = splits[0:-1]

			for split in splits:
				right_inds, left_inds = self.split(data, split_rule=(feature,split))

				#produce hist
				right_labels = labels[right_inds]
				left_labels = labels[left_inds]
				
				right_hist = [np.sum(right_labels==i) for i in np.unique(right_labels)]
				left_hist = [np.sum(left_labels==i) for i in np.unique(left_labels)]
				
		
				#for each split - measure badness
				impurity = self.impurity(left_hist, right_hist)

				#save split info for lowest impuriy (most info gain)
				if impurity < min_impurity:
					min_impurity = impurity
					split_feature = feature
					split_criteria = split #split if > then this value
		
		# if no further progress can be made, return node as leaf node.
		if min_impurity == 1.:
			return None
			# return Node(label = node_label, size=len(data))
			
		# otherwise return node with split info.
		self.split_rule = (split_feature, split_criteria)


		# return Node((split_feature, split_criteria), size=len(data), label=node_label) 

	def impurity(self, left_label_hist, right_label_hist):
		'''A method that takes in the result of a split: two histograms (a histogram 
		is a mapping from label values to their frequencies) that count the frequencies
		of labels on the "left" and "right" side of that split. The method calculates 
		and outputs a scalar value representing the impurity (i.e. the "badness") of 
		the specified split on the input data.'''
		sl,sr = left_label_hist,right_label_hist
		# weighted value
		impurity = (sum(sl)*self.cost_Js(sl)+sum(sr)*self.cost_Js(sr))/(sum(sl)+sum(sr))
		return impurity
	

	def cost_Js(self, s_hist):
		#measure entropy
		entropy = 0
		for c in s_hist:
			pc = float(c)/sum(s_hist) #count of c/number of samples in set.
			entropy += -pc*np.log2(pc)
		return entropy


		
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
	