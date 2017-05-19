import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pandas as pd
from collections import defaultdict

# Fast order-preserving elimination of duplicates
def f7(seq):
	seen = set()
	return [ x for x in seq if x not in seen and not seen.add(x)]

def maxFreq(L, default):
	"""
	Return a tuple (a, b): a is the maximum occuring element and b is its frequency.
	If all elements of dict are same then return the value at `default` position of list.
	This `default` position contains the prediction of the maximum precision classifier.
	"""
	d = defaultdict(int)
	for i in L:
		d[i] += 1
	values = d.values()
	if values.count(d[0])==len(d):
		return (L[default], d[0])
	result = max(d.iteritems(), key=lambda x: x[1])
	return result

def extractDataAndTargets():
	"""
	This function requires appropriate value of datasetPath to be set before being called.
	Example values:
		datasetPath = 'iris_new.csv'
	This reads the file in `datasetPath` and returns the rows and the targets of the data.
		"""
	data = pd.read_csv(datasetPath, sep=',').values

	## NOTE: Extract the classes of each row from the first/last column.
	targets = [data[x][0] for x in xrange(0, len(data))]
	classes = f7(targets)
	# Total number of classes in the dataset.
	noOfTotalClasses = len(classes)
	# Total number of vectors available for one class in the dataset.
	noOfTotalVectors = len(data)/noOfTotalClasses

	## NOTE: Remove extraneous data here. Remove the class, sessionIndex and rep of the row.
	data = data[:, 3:]

	mapClasses = {}
	for a,b in enumerate(list(classes)):
		mapClasses[b] = a
	mapTargets = np.array([mapClasses[target] for target in targets])
	return data, mapTargets

noOfTotalClasses = 12
# Total number of vectors available for one class.
noOfTotalVectors = 150
# For training purposes for one class use first `noOfTrainingVectors` vectors.
noOfTrainingVectors = 100
# For testing purposes for one class use first `noOfTestingVectors` vectors.
# noOfTestingVectors = noOfTotalVectors - noOfTrainingVectors
noOfTestingVectors = 50

noOfFeatures = 31
n_splitsForGridSearch = 9
trainingRatio = 0.64
testingRatio = 1 - trainingRatio
# This contains the path for the dataset.
datasetPath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
datasetPath = datasetPath + os.sep + "OURData.csv"

noOfInputNodes = noOfFeatures
# The number of Hidden nodes is taken as (2*P)/3, where P is the number of the input nodes
noOfHiddenNodes = 15
# The number of output nodes is equal to the number of classes
noOfOutputNodes = noOfTotalClasses


X, y = extractDataAndTargets()
# Provides train/test indices to split data in train/test sets. The folds are made by preserving the percentage of samples for each class.
ss = StratifiedShuffleSplit(n_splits=1, test_size=testingRatio, random_state=0)

for train_index, test_index in ss.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

# sgd_clf = MLPClassifier(hidden_layer_sizes = (noOfInputNodes, noOfHiddenNodes, noOfOutputNodes), 
# 		activation = "tanh", solver = "sgd", max_iter = 1800, learning_rate = "adaptive", learning_rate_init=0.01)
# adam_clf = MLPClassifier(hidden_layer_sizes = (noOfInputNodes, noOfHiddenNodes, noOfOutputNodes), 
# 		activation = "tanh", solver = "adam", max_iter = 1000)

# Set the parameters by cross-validation

# learning_rates = list(np.arange(0.0001,0.015,0.005))
learning_rates = [0.005, 0.01]
hidden_nodes = list(np.arange(15,25,1))
alpha = 10.0**-np.arange(1,7)

# array_hidden = np.zeros(shape=[len(hidden_nodes), 3])
array_hidden = []
for i,x in enumerate(hidden_nodes):
	array_hidden.append((31, x, 12))

# array_hidden = np.array(array_hidden, dtype="int32")

tuned_parameters_sgd = [{'solver' : ['sgd'], 'learning_rate_init' : learning_rates, 'learning_rate' : ['adaptive'], 
						'hidden_layer_sizes' : array_hidden, 'alpha' : alpha}]
tuned_parameters_adam = [{'solver' : ['adam'], 'learning_rate_init' : learning_rates, 
						'hidden_layer_sizes' : array_hidden, 'alpha' : alpha}]
# tuned_parameters_adam = [{'kernel': ['rbf'], 'gamma': gamma_parameters, 'C': C_parameters}]
# tuned_parameters_sgm = [{'kernel': ['sigmoid'], 'gamma': gamma_parameters, 'C': C_parameters}]
y_pred = []
score = 'precision_macro'
for i, tuned_parameters in enumerate([tuned_parameters_sgd, tuned_parameters_adam]):
	print("# Tuning hyper-parameters for %s" % tuned_parameters)
	print()

	cv = StratifiedKFold(n_splits=n_splitsForGridSearch, shuffle=True, random_state=0)
	clf = GridSearchCV(MLPClassifier(activation='tanh', max_iter=1800), tuned_parameters, cv=cv, scoring=score)
	clf.fit(X_train, y_train)

	print("Grid scores on training set:")
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
	print()
	print("Best parameters set found on training set:")
	print(clf.best_params_)

	print("Detailed classification report:")
	print("The model is trained on the full training set. The scores are computed on the full evaluation set.")
	y_true, y_pred = y_test, clf.predict(X_test)
	print(classification_report(y_true, y_pred))
# This list contains a sub-list for every testing vector. This sub-list contains the predictions by every classifier.
# Example: [[2, 3, 1], [0, 0, 0], [2, 2, 2], [2, 2, 2], [1, 1, 1]] for 5 testing vectors.
# y_comb = [ [y_pred[y][x] for y in xrange(0, 3)] for x in xrange(0, len(y_pred[0])) ]
# y_final = []

# print "Predictions Actual-User Best-Prediction"
# for i in xrange(0, len(y_comb)): print y_comb[i], y_test[i], maxFreq(y_comb[i], default=0)[0]

# for i in xrange(0, len(y_comb)):
# 	y_final.append( maxFreq(y_comb[i], default=0)[0] )
# combinedResults = precision_recall_fscore_support(y_true, y_final, average='macro')
# print combinedResults