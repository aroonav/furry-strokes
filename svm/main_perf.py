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

Optimal value for TPR and TNR is 1.
Optimal value for FPR and FNR is 0.
=================================
"""
print(__doc__)

import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import itertools
from sklearn import svm
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
													normalize=False,
													title='Confusion matrix',
													cmap=plt.cm.Blues):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		if normalize:
				cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
				print("Normalized confusion matrix")
		else:
				print('Confusion matrix, without normalization')

		print(cm)

		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
				plt.text(j, i, round(cm[i, j], 2),
								 horizontalalignment="center",
								 color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout(pad=0.02, h_pad=None, w_pad=None, rect=None)
		plt.ylabel('True class')
		plt.xlabel('Predicted class')

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
		# if i==0:
		# 	for j in range(noOfTotalVectors):
		# 		tempData = reader.next()						# Discard one vector
		# 	continue
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
		# if i==0:
		# 	for j in range(noOfTotalVectors):
		# 		tempData = reader.next()						# Discard one vector
		# 	continue
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

def showCnfValues():
	columnNames = ["TP", "FN", "FP", "TN", "TPR", "TNR", "FNR", "FPR"]
	# Each row denotes a class. Each column contains TP, FN, FP, TN, TPR, TNR, FNR, FPR values.
	cnf_values = [[0 for x in range(len(columnNames))] for x in range(noOfTotalClasses)]
	total = [0 for x in range(8)]
	# TODO: Improve the efficiency of the calculation of these values. It's O(n^3) now.
	# For each class iterate through the entire confusion matrix, calculate and save the values in the matrix.
	for i in xrange(0, noOfTotalClasses):
		for j in xrange(0, noOfTotalClasses):
			for k in xrange(0, noOfTotalClasses):
				if j!=k and k==i :		# FP Values
					cnf_values[i][2] += cnf_matrix[j][k]
				if j==k and j==i:		# TP Values
					cnf_values[i][0] = cnf_matrix[j][k]
				if j!=k and j==i:		# FN values
					cnf_values[i][1] += cnf_matrix[j][k]
				if j!=i and k!=i:								# TN values
					cnf_values[i][3] += cnf_matrix[j][k]
	for i in xrange(0, noOfTotalClasses):
		cnf_values[i][4] = round(cnf_values[i][0]/float(cnf_values[i][0]+cnf_values[i][1]), 3) 		# TPR = TP/(TP+FN)
		cnf_values[i][5] = round(cnf_values[i][3]/float(cnf_values[i][3]+cnf_values[i][2]), 3) # TNR = TN/(TN+FP)
		cnf_values[i][6] = 1-cnf_values[i][4] 		# FNR = FN/(FN+TP) = 1-TPR
		cnf_values[i][7] = 1-cnf_values[i][5] 		# FPR = FP/(FP+TN) = 1-TNR
	print "Values from confusion matrix:"
	rowNames = class_names
	row_format ="{:>6}" * (len(columnNames) + 1)
	print row_format.format("", *columnNames)
	for rowName, row in zip(rowNames, cnf_values):
		print row_format.format(rowName, *row)

	for i in xrange(0, 4):
		total[i] = sum([row[i] for row in cnf_values])
	total[4] = round(total[0]/float(total[0]+total[1]), 3) 		# TPR = TP/(TP+FN)
	total[5] = round(total[3]/float(total[3]+total[2]), 3)		# TNR = TN/(TN+FP)
	total[6] = round(1-total[4], 3) 		# FNR = FN/(FN+TP) = 1-TPR
	total[7] = round(1-total[5], 3) 		# FPR = FP/(FP+TN) = 1-TNR
	print row_format.format("Total", *total)

# NOTE: Change the value of noOfTotalClasses, noOfTrainingVectors
# and noOfTestingVectors in actual use.
# Total number of classes.
noOfTotalClasses = 3
# Total number of vectors available for one class.
noOfTotalVectors = 400
# For training purposes for one class use first `noOfTrainingVectors` vectors.
noOfTrainingVectors = 20
# For testing purposes for one class use first `noOfTestingVectors` vectors.
noOfTestingVectors = noOfTotalVectors - noOfTrainingVectors
# Each vector contains `noOfFeatures` features.
noOfFeatures = 31
# This contains the no of classifiers defined below
noOfClassifiers = 5
# This contains the path for the dataset.
datasetPath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
datasetPath = datasetPath + os.sep + "DSL-StrongPasswordData.csv"
max_training = 0
max_scores = []
# Maximum accuracy of each classifier
max_perf_classifiers = [0]*5
# Amount of training data required to achieve maximum accuracy
max_perf_trainingData = [0]*5
trainingData_start = 2
trainingData_end = trainingData_start + noOfTrainingVectors

performanceMat = [[0 for x in range(noOfTrainingVectors)] for x in range(5)]
# Required for the confusion matrix
# max_Z contains the predicted values when the accuracy is maximum
max_Z = [[0 for x in range(noOfTestingVectors)] for x in range(5)]
max_test_y = [[0 for x in range(noOfTestingVectors)] for x in range(5)]
# TODO: Automate this
class_names = ['s002', 's003', 's004']


# Iterate over the all possible amounts of training vectors.
for x in xrange(trainingData_start, trainingData_end):
	noOfTrainingVectors = x
	noOfTestingVectors = noOfTotalVectors - noOfTrainingVectors
	# X: We take all the features. Or we can take only some features here by slicing.
	# y: This contains the actual classes for each training vector i.e the target.
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
			max_Z[i] = Z
			max_test_y[i] = test_y


# ==========================
# MEASUREMENT OF PERFORMANCE
# ==========================
# Measure 1: Maximum accuracy of the model over the given training size
# defined by trainingData_start and trainingData_end
print "Maximum accuracy(from 0 to 1) of the 5 classifiers is: ", max_perf_classifiers
print "Maximum accuracy is attained for training size: ", max_perf_trainingData, "\n"


# Measure 2: Plot the performance vs training size.
# Scatter plot the maximum (accuracy, training size) points of each classifier
labels = ['SVC(linear)', 'LinearSVC', 'SVC(rbf)', 'SVC(poly)', 'NuSVC(rbf)']
for x in xrange(0, 5):
	plt.plot(range(trainingData_start, trainingData_end), performanceMat[x], label=labels[x])
	# Plot the maximum accuracy points
plt.scatter(max_perf_trainingData, max_perf_classifiers)
	# Annotate the maximum accuracy points
for i in xrange(0, 5):
	pt_label = "(" + str(max_perf_trainingData[i]) + "," + "{0:.3f}".format(max_perf_classifiers[i]) + ")"
	plt.annotate(pt_label, xy=(max_perf_trainingData[i], max_perf_classifiers[i]),
								xytext=(4,4), textcoords='offset points', fontsize='x-small')
#TODO: Point annotations are overlapping. Correct it.
legend = plt.legend(loc = 'upper left', shadow=True, fontsize=8)
legend.get_frame()
plt.title("Graph 1: Accuracy vs Training size")
plt.xlabel("Training size")
plt.ylabel("Accuracy(0.0-1.0)")


# Measure 3: Plot the confusion matrix for each classifier when the accuracy is the most.
plt.figure(figsize = (11,5))
np.set_printoptions(precision=3)
for i in xrange(0,5):
	print "Analysis of classifier", labels[i]
	print "================================="
	cnf_matrix = confusion_matrix(max_test_y[i], max_Z[i])
	# Plot non-normalized confusion matrix
	plt.subplot(2,5,i+1)
	plot_confusion_matrix(cnf_matrix, classes=class_names,
												title=labels[i])
	# Plot normalized confusion matrix
	plt.subplot(2,5,i+6)
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
												title=labels[i])
	showCnfValues()
	print "\n"
plt.show()


# TODO: Measure 4: Plot the performance of each classifier for higher no. of classes when training size is set at the point of maximum accuracy for 3 classes.
