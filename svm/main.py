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
# import matplotlib.pyplot as plt
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
# This contains the path for the dataset.
datasetPath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
datasetPath = datasetPath+os.sep+"DSL-StrongPasswordData.csv"

# X: We take all the features. Or we can take only some features here by slicing.
# y: This contains the actual classes for each training vector i.e the target.
X,y = load_trainingData()
test_X,test_y = load_testingData()

# print X,y,"\n"
# print "\n"
# print test_X,test_y,"\n"


# we create instances of SVM and fit our data.
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)
nu_svc = svm.NuSVC().fit(X, y) 

for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc, nu_svc)):
	if(i==4):
		print "Testing with the RBF kernel in svm.NuSVC now."
	elif(i!=1):
		print "Testing with the", clf.kernel, "classsifier now."
	else:
		print "Testing with the linearSVC classsifier now."	
	# Pass testing data to the classifier
	Z = clf.predict(test_X)
	# print Z
	similar = 0
	for j in xrange(0, noOfTestingVectors*noOfTotalClasses):
		# print j, Z[j], test_y[j]
		if Z[j]==test_y[j]:
			similar+=1
	score = (similar/float(noOfTestingVectors*noOfTotalClasses))*100
	print "Performance of classifier:", score,"%"
	print ""
