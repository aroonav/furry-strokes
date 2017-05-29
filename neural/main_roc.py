"""
============================================
Generate ROC Curves for all the classifiers.
============================================
We consider all the 31 features for classification.
For more details about IIITBh-keystroke dataset goto: https://github.com/aroonav/IIITBh-keystroke
For more details about CMU's dataset goto: http://www.cs.cmu.edu/~keystroke/
These features are used to train and test NN & SVM models for
classifying users on the basis of the timings between their keystrokes.

This will generate the ROC Curves.
"""
# print(__doc__)

import os
import numpy as np
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import svm


# Fast order-preserving elimination of duplicates
def removeDupes(seq):
	seen = set()
	return [ x for x in seq if x not in seen and not seen.add(x)]

def convert_numerical_train(labels):
	x = []
	for i in range(noOfTotalClasses):
		for j in range(noOfTrainingVectors):
			# x.append(int(labels[(i*noOfTotalVectors) + j][-2:]))
			x.append(i)
	return x

def convert_numerical_test(labels):
	x = []
	for i in range(noOfTotalClasses):
		for j in range(noOfTestingVectors):
			# x.append(int(labels[(i*noOfTotalVectors) + j][-2:]))
			x.append(i)
	return x

def load_trainingData():
	"""
	This reads file DSL-StrongPasswordData.csv and returns the training data in
	an ndarray of shape noOfTrainingVectors*noOfFeatures and target ndarray
	of shape (noOfTrainingVectors*noOfTotalClasses)*1.
	"""
	global datasetPath
	dataset = np.empty([0,noOfFeatures])
	target = np.empty(0)
	file = open(datasetPath)
	reader = csv.reader(file)
	reader.next()
	for i in range(noOfTotalClasses):
	# for i in range(noOfTotalClasses+1):
	# 	if i == 0:
	# 		for j in xrange(noOfTotalVectors):
	# 			reader.next()
	# 		continue
		for j in range(noOfTrainingVectors):
			tempData = reader.next()					# Read one vector
			currentSubject = tempData[0]			# Save subject's name
			for k in range(3):								# Discard first 3 values
				del tempData[0]
			tempData = map(float, tempData)
			tempData = np.array(tempData, ndmin=2)
			dataset = np.append(dataset, tempData, axis=0)
			target  = np.append(target, [currentSubject], axis=0)
		for j in range(noOfTestingVectors):	# Discard testing vectors for now
			tempData = reader.next()					# Discard one vector
		# Discard the rest of the unused vectors now
		for j in range(noOfTotalVectors - noOfTrainingVectors - noOfTestingVectors):
			tempData = reader.next()						# Discard one vector
	return dataset,target

def load_testingData():
	"""
	TODO: Merge load_testingData() and load_trainingData() functions
	This reads file DSL-StrongPasswordData.csv and returns the testing data in
	an ndarray of shape noOfTestingVectors*noOfFeatures and target ndarray
	of shape (noOfTestingVectors*noOfTotalClasses)*1.
	"""
	global datasetPath
	dataset = np.empty([0,noOfFeatures])
	target = np.empty(0)
	file = open(datasetPath)
	reader = csv.reader(file)
	reader.next()
	for i in range(noOfTotalClasses):
	# for i in range(noOfTotalClasses+1):
	# 	if i == 0:
	# 		for j in xrange(noOfTotalVectors):
	# 			reader.next()
	# 		continue
		for j in range(noOfTrainingVectors):	# Discard training vectors now
			tempData = reader.next()						# Discard one vector
		for j in range(noOfTestingVectors):
			tempData = reader.next()						# Read one vector
			currentSubject = tempData[0]				# Save subject's name
			for k in range(3):									# Discard first 3 values
				del tempData[0]
			tempData = map(float, tempData)
			tempData = np.array(tempData, ndmin=2)
			dataset = np.append(dataset, tempData, axis=0)
			target = np.append(target, [currentSubject], axis=0)
		# Discard the rest of the unused vectors now
		for j in range(noOfTotalVectors - noOfTrainingVectors - noOfTestingVectors):
			tempData = reader.next()						# Discard one vector
	return dataset,target


# Total number of classehidden_layer_sizes.
noOfTotalClasses = 5
# Total number of vectors available for one class.
noOfTotalVectors = 250
# For training purposes for one class use first `noOfTrainingVectors` vectors.
noOfTrainingVectors = 96
# For testing purposes for one class use first `noOfTestingVectors` vectors.
# noOfTestingVectors = noOfTotalVectors - noOfTrainingVectors
noOfTestingVectors = 154
# Each vector contains `noOfFeatures` features.
noOfFeatures = 31
# This contains the path for the dataset.
datasetPath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
# datasetPath = datasetPath + os.sep + "DSL-StrongPasswordData.csv"
datasetPath = datasetPath + os.sep + "IIITBh-Small.csv"


noOfInputNodes = noOfFeatures
# The number of Hidden nodes is taken as (2*P)/3, where P is the number of the input nodes
noOfHiddenNodes = 15
# The number of output nodes is equal to the number of classes
noOfOutputNodes = noOfTotalClasses

# X: We take all the features. Or we can take only some features here by slicing.
# y: This contains the actual classes for each training vector i.e the target.
X,y = load_trainingData()
test_X,test_y = load_testingData()
classes = removeDupes(y)

y = convert_numerical_train(y)
test_y = convert_numerical_test(test_y)

y = np.array(y)
test_y = np.array(test_y)

# binarize output labels
y_binarized = label_binarize(y, classes=range(noOfTotalClasses))
test_y_binarized = label_binarize(test_y, classes=range(noOfTotalClasses))

# Neural Classifiers
sgd_clf = MLPClassifier(hidden_layer_sizes = (noOfInputNodes, noOfHiddenNodes, noOfOutputNodes), 
		activation = "tanh", solver = "sgd", max_iter = 1800, learning_rate = "adaptive", learning_rate_init="0.01",
		random_state=0)
adam_clf = MLPClassifier(hidden_layer_sizes = (noOfInputNodes, noOfHiddenNodes, noOfOutputNodes), 
		activation = "tanh", solver = "adam", max_iter = 1000, random_state=0)
# SVM Classifiers
rbf_svc_clf = OneVsRestClassifier((svm.SVC(kernel='rbf', gamma=0.05, C=401, probability=True)))
lin_svc_clf = OneVsRestClassifier((svm.SVC(kernel='linear', C=801, gamma=0.01, probability=True)))

sgd = sgd_clf.fit(X,y_binarized)
adam = adam_clf.fit(X,y_binarized)
lin_svc = lin_svc_clf.fit(X, y_binarized)
rbf_svc = rbf_svc_clf.fit(X, y_binarized)

labels = ['Neural(SGD)', 'Neural(adam)', 'SVC(linear)', 'SVC(rbf)']
colors = ['black', 'blue', 'darkorange', 'violet', 'yellow', 'red', 'pink', 'green', 'magenta', 'cyan', 'grey', 'brown']

for i, clf in enumerate((sgd, adam, lin_svc, rbf_svc)):
	y_score = clf.predict_proba(test_X)
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for j in range(noOfTotalClasses):
		fpr[j], tpr[j], thresholds = roc_curve(test_y_binarized[:, j], y_score[:, j])
		roc_auc[j] = auc(fpr[j], tpr[j])

	plt.figure()
	lw = 2
	for j,color in enumerate(colors[:noOfTotalClasses]):
		plt.plot(fpr[j], tpr[j], color=color,
		         lw=lw, label=classes[j]+' (area = %0.2f)' % roc_auc[j])

	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic for '+labels[i])
	plt.legend(loc="lower right")
plt.show()
