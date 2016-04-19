# Decision tree

DecisionTree implement Decision Tree Classification. It has 3 main methods - train, predict and prune. Prune is an optional step in the training process. Given a pruning data set and pruning labels it will try to identify cases of overfitting by eliminating splits that do not increase the classification rate on a holdout pruning data ser.

Decision tree takes the following optional arguments:

	min_leaf = minimum leaf size. Return a leaf node if a node size is less than or equal to min_leaf
	max_depth = maximum depth of the tree
	m = (for Random Forest Primarily) number of randomly selected features to use at each split.


Methods:

	train(data,labels) - returns root node of decision tree
	predict(data) - returns data labels
	prune(data,labels) - creates new leaf by deleting splits if misclassification is not improved on holdout set. returns labels and number of pruning data points misclassified with pruned tree.


#Random Forest

Random Forest creates multiple random trees, and for each training point assigns to class 1 or zero based on which class the data point was assigned to most frequently.

Random forest takes the following required arguments:
	m = number of random features to test at each split.
	iteration = number of trees to include in the random forest

And the following optional arguments:
	min_leaf = minimum leaf size for each tree in the forest
	max_depth = maximum depth of each tree in the forest

Methods:
	train(data,labels) - returns root node of decision tree
	predict(data) - returns data labels