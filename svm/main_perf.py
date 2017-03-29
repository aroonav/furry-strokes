"""
=================================
SVM Models for keystroke dynamics
=================================

Implementation and comaprison of performance of four SVM classifiers(1 is
 ``LinearSVC()`` and the others are ``SVC()``).

The linear models ``LinearSVC()`` and ``SVC(kernel='linear')`` yield slightly
different results. This can be a consequence of the following differences:

- ``LinearSVC`` minimizes the squared hinge loss while ``SVC`` minimizes the
  regular hinge loss.
- ``LinearSVC`` uses the One-vs-All (also known as One-vs-Rest) multiclass
  reduction while ``SVC`` uses the One-vs-One multiclass reduction.

We consider all the 31 features for classification. For more details about
the dataset in DSL-StrongPasswordData.csv, goto: http://www.cs.cmu.edu/~keystroke/

These features are used to train and test SVM models for classifying users on
the basis of the timings between their keystrokes.
"""
print(__doc__)

import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import svm

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


# NOTE: Change the value of noOfTotalClasses, noOfTrainingVectors and noOfTestingVectors in actual use.
# Total number of classes.
noOfTotalClasses = 3
# Total number of vectors available for one class.
noOfTotalVectors = 400
# For training purposes for one class use first `noOfTrainingVectors` vectors.
noOfTrainingVectors = 250
# For testing purposes for one class use first `noOfTestingVectors` vectors.
# noOfTestingVectors = noOfTotalVectors - noOfTrainingVectors
noOfTestingVectors = 100
# Each vector contains `noOfFeatures` features.
noOfFeatures = 31
# This contains the no of classifiers defined below
noOfClassifiers = 5
# This contains the path for the dataset.
datasetPath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
datasetPath = datasetPath + os.sep + "DSL-StrongPasswordData.csv"

# X: We take all the features. Or we can take only some features here by slicing.
# y: This contains the actual classes for each training vector i.e the target.

max_training = 0
max_scores = []
max_perf_classifiers = [0]*5
max_perf_trainingData = [0]*5

trainingData_start = 2
trainingData_end = 350
performanceMat = [[0 for x in range(trainingData_end-trainingData_start)] for x in range(5)]

# Iterate over the all possible amounts of training vectors.
for x in xrange(trainingData_start, trainingData_end):
	noOfTrainingVectors = x
	noOfTestingVectors = noOfTotalVectors - noOfTrainingVectors
	X,y = load_trainingData()
	test_X,test_y = load_testingData()

	# we create instances of SVM and fit our data.
	C = 1.0  # SVM regularization parameter
	svc = svm.SVC(kernel='linear', C=C).fit(X, y)
	lin_svc = svm.LinearSVC(C=C).fit(X, y)
	rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
	poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
	nu_svc = svm.NuSVC().fit(X, y) 

	for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc, nu_svc)):
		# Pass testing data to the classifier
		Z = clf.predict(test_X)
		similar = 0
		for j in xrange(0, noOfTestingVectors*noOfTotalClasses):
			if Z[j]==test_y[j]:
				similar+=1
		score = (similar/float(noOfTestingVectors*noOfTotalClasses))
		performanceMat[i][x-trainingData_start] = score
		if max_perf_classifiers[i]<score:
			max_perf_classifiers[i] = score
			max_perf_trainingData[i] = noOfTrainingVectors


# Measure 1: Maximum performance of the model over the given training size defined by trainingData_start and trainingData_end
print "Maximum accuracy(from 0 to 1) of the 5 classifiers is: ", max_perf_classifiers
print "Maximum accuracy is attained for training size: ", max_perf_trainingData
print ""

# Measure 2:
labels = ['SVC(linear)', 'LinearSVC', 'SVC(rbf)', 'SVC(poly)', 'NuSVC(rbf)']
for x in xrange(0, 5):
	plt.plot(range(trainingData_start, trainingData_end), performanceMat[x], label=labels[x])
legend = plt.legend(loc = 'upper left', shadow=True, fontsize='x-large')
legend.get_frame()
plt.xlabel("Training size")
plt.ylabel("Accuracy(0.0-1.0)")
plt.show()
