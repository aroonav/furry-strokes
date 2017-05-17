"""
=======================================================================
Parameter tuning using Grid Search of SVM Models for keystroke dynamics
=======================================================================
We consider all the 31 features for classification.
For more details about IIITBh-keystroke dataset goto: https://github.com/aroonav/IIITBh-keystroke
For more details about CMU's dataset goto: http://www.cs.cmu.edu/~keystroke/
These features are used to train and test NN & SVM models for
classifying users on the basis of the timings between their keystrokes.

This will find the parameters for the best performing classifier when
Precision is considered as the metric using Grid Search. It will also print all 
the grid scores.
It will then print these best parameters and will print the Precision along
with a classification report consisting of Precision, recall, F1-Score and support.
"""
# print(__doc__)

import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pandas as pd
from collections import defaultdict

def f7(seq):
	"""
	Fast order-preserving elimination of duplicates
	"""
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


noOfFeatures = 31
n_splitsForGridSearch = 12
trainingRatio = 0.64
testingRatio = 1 - trainingRatio
# This contains the path for the dataset.
fileName = "IIITBh-Small.csv"
datasetPath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
datasetPath = datasetPath + os.sep + fileName

X, y = extractDataAndTargets()
# Provides train/test indices to split data in train/test sets. The folds are made by preserving the percentage of samples for each class.
ss = StratifiedShuffleSplit(n_splits=1, test_size=testingRatio, random_state=0)

for train_index, test_index in ss.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

# Set the parameters by cross-validation
gamma_parameters = [1e-2, 1e-1, 1e-0, 5e-2, 5e-1, 5e-0]
C_parameters = list(np.arange(1, 1000, 100))
tuned_parameters_lin = [{'kernel': ['linear'], 'gamma': gamma_parameters, 'C': C_parameters}]
tuned_parameters_rbf = [{'kernel': ['rbf'], 'gamma': gamma_parameters, 'C': C_parameters}]
# tuned_parameters_sgm = [{'kernel': ['sigmoid'], 'gamma': gamma_parameters, 'C': C_parameters}]
score = 'precision_macro'
for i, tuned_parameters in enumerate([tuned_parameters_lin, tuned_parameters_rbf]):
	print("# Tuning hyper-parameters for %s" % tuned_parameters)
	print()

	cv = StratifiedKFold(n_splits=n_splitsForGridSearch, shuffle=True, random_state=0)
	clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=cv, scoring=score)
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
