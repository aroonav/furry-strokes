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
Example of cnf_values[]:
Values from confusion matrix:
	 Subject  TP    FN      FP    TN   TPR    TNR    FNR    FPR       Accuracy
	 s002     21    359      8    752  0.055  0.989  0.945  0.011     0.678
	 s003    355     25    230    530  0.934  0.697  0.066  0.303     0.776
	 s004    236    144    290    470  0.621  0.618  0.379  0.382     0.619
	Total    612    528    528   1752  0.537  0.768  0.463  0.232     0.691
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

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

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

def getClassNames():
	"""
	This retrieves the names of the subjects/classes
	"""
	class_names = []
	file = open(datasetPath)
	reader = csv.reader(file)
	reader.next()
	for i in range(noOfTotalClasses):
		tempData = reader.next()					# Read one vector
		currentSubject = tempData[0]			# Save subject's name
		class_names.append(currentSubject)
		for j in range(noOfTotalVectors-1):
			reader.next()			# Discard one vector
	return class_names

def load_trainingData(tempTrainingVectors, tempTestingVectors):
	"""
	This reads file DSL-StrongPasswordData.csv and returns the training data in
	an ndarray of shape tempTrainingVectors*noOfFeatures and target ndarray
	of shape (tempTrainingVectors*noOfTotalClasses)*1.
	"""
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
		for j in range(tempTrainingVectors):
			tempData = reader.next()					# Read one vector
			currentSubject = tempData[0]			# Save subject's name
			for k in range(3):								# Discard first 3 values
				del tempData[0]
			tempData = map(float, tempData)
			tempData = np.array(tempData, ndmin=2)
			dataset = np.append(dataset, tempData, axis=0)
			target  = np.append(target, [currentSubject], axis=0)
		for j in range(tempTestingVectors):	# Discard testing vectors for now
			tempData = reader.next()					# Discard one vector
		# Discard the rest of the unused vectors now
		for j in range(noOfTotalVectors - tempTrainingVectors - tempTestingVectors):
			tempData = reader.next()						# Discard one vector
	return dataset,target

def load_testingData(tempTrainingVectors, tempTestingVectors):
	"""
	TODO: Merge load_testingData() and load_trainingData() functions
	This reads file DSL-StrongPasswordData.csv and returns the testing data in
	an ndarray of shape tempTestingVectors*noOfFeatures and target ndarray
	of shape (tempTestingVectors*noOfTotalClasses)*1.
	"""
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
		for j in range(tempTrainingVectors):	# Discard training vectors now
			tempData = reader.next()						# Discard one vector
		for j in range(tempTestingVectors):
			tempData = reader.next()						# Read one vector
			currentSubject = tempData[0]				# Save subject's name
			for k in range(3):									# Discard first 3 values
				del tempData[0]
			tempData = map(float, tempData)
			tempData = np.array(tempData, ndmin=2)
			dataset = np.append(dataset, tempData, axis=0)
			target = np.append(target, [currentSubject], axis=0)
		# Discard the rest of the unused vectors now
		for j in range(noOfTotalVectors - tempTrainingVectors - tempTestingVectors):
			tempData = reader.next()						# Discard one vector
	return dataset,target

def fillCnfValues(cnf_matrix):
	"""
	This function saves the TP, FN, FP, TN, TPR, TNR, FNR, FPR, Accuracy values in the matrix
	of dimensions [len(columnNames), noOfTotalClasses] and returns it.
	"""
	# TODO: Improve the efficiency of the calculation of these values. It's O(n^3) now.
	# For each class iterate through the entire confusion matrix, calculate and save the values in the matrix.
	total = [0 for x in range(len(columnNames))]
	cnf_values = [[0 for x in range(len(columnNames))] for x in range(noOfTotalClasses+1)]
	for i in xrange(0, noOfTotalClasses):
		for j in xrange(0, noOfTotalClasses):
			for k in xrange(0, noOfTotalClasses):
				if j!=k and k==i :	# FP Values
					cnf_values[i][2] += cnf_matrix[j][k]
				if j==k and j==i:		# TP Values
					cnf_values[i][0] = cnf_matrix[j][k]
				if j!=k and j==i:		# FN values
					cnf_values[i][1] += cnf_matrix[j][k]
				if j!=i and k!=i:		# TN values
					cnf_values[i][3] += cnf_matrix[j][k]
	for i in xrange(0, noOfTotalClasses):
		cnf_values[i][4] = round(cnf_values[i][0]/float(cnf_values[i][0]+cnf_values[i][1]), 3) # TPR = TP/(TP+FN)
		cnf_values[i][5] = round(cnf_values[i][3]/float(cnf_values[i][3]+cnf_values[i][2]), 3) # TNR = TN/(TN+FP)
		cnf_values[i][6] = round(1-cnf_values[i][4], 3) 		# FNR = FN/(FN+TP) = 1-TPR
		cnf_values[i][7] = round(1-cnf_values[i][5], 3) 		# FPR = FP/(FP+TN) = 1-TNR
		cnf_values[i][8] = round((cnf_values[i][0]+cnf_values[i][3])/float(cnf_values[i][0]+cnf_values[i][1]+cnf_values[i][2]+cnf_values[i][3]), 3) # Accuracy = (TP+TN)/(TP+FN+FP+TN)
	for i in xrange(0, 4):
		total[i] = sum([row[i] for row in cnf_values])
	total[4] = round(total[0]/float(total[0]+total[1]), 3) 		# TPR = TP/(TP+FN)
	total[5] = round(total[3]/float(total[3]+total[2]), 3)		# TNR = TN/(TN+FP)
	total[6] = round(1-total[4], 3) 		# FNR = FN/(FN+TP) = 1-TPR
	total[7] = round(1-total[5], 3) 		# FPR = FP/(FP+TN) = 1-TNR
	total[8] = round( (total[0]+total[3])/float(total[0]+total[1]+total[2]+total[3]), 3)

	cnf_values[-1] = total
	return cnf_values

def showCnfValues(cnf_matrix):
	"""
	This prints the TP, FN, FP, TN, TPR, TNR, FNR, FPR, Accuracy values and color codes the important values.
	It also prints the overall performance of the classifier(present in total[] matrix)
	"""
	# Each row denotes a class. Each column contains TP, FN, FP, TN, TPR, TNR, FNR, FPR, Accuracy values.
	cnf_values = fillCnfValues(cnf_matrix)

	print "Values from confusion matrix:"
	rowNames = class_names
	row_format = "{:>7}" * (len(columnNames)) + "{:>10}"
	print row_format.format("", *columnNames)
	for rowName, row in zip(rowNames, cnf_values):
		print row_format.format(rowName, *row)

	row_format1 = "{:>7}"*5
	row_format2 = "{:>5}" + "{:>7}"
	row_format3 = "{:>8}"
	total = cnf_values[-1]
	print row_format1.format("Total", *(total[:4])),
	print bcolors.OKGREEN,
	print row_format2.format(*(total[4:6])),
	print bcolors.ENDC,
	print row_format2.format(*(total[6:8])),
	print bcolors.OKGREEN,
	print row_format3.format(total[8])
	print bcolors.ENDC,

def trainAndTest(C):
	"""
	This will train the SVM Models and then test the models.
	C is the SVM regularization parameter
	"""
	# Training phase
	global TPRMatrix,accuracyMatrix , max_TPR_classifiers,max_accuracy_classifiers, trainingDataForMaxTPR,trainingDataForMaxAccuracy, maxTPR_Z,maxAcc_Z, maxTPR_test_y,maxAcc_test_y
	# Iterate over the all possible amounts of training vectors.
	for x in xrange(trainingData_start, trainingData_end):
		tempTrainingVectors = x
		tempTestingVectors = noOfTotalVectors - tempTrainingVectors
		# X: We take all the features. Or we can take only some features here by slicing.
		# y: This contains the actual classes for each training vector i.e the target.
		X,y = load_trainingData(tempTrainingVectors, tempTestingVectors)
		test_X,test_y = load_testingData(tempTrainingVectors, tempTestingVectors)

		# we create instances of SVM and fit our data.
		svc = svm.SVC(kernel='linear', C=C).fit(X, y)
		lin_svc = svm.LinearSVC(C=C).fit(X, y)
		rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
		poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
		nu_svc = svm.NuSVC().fit(X, y) 

		for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc, nu_svc)):
			# Pass testing data to the classifier
			Z = clf.predict(test_X)
			cnf_matrix = confusion_matrix(test_y, Z)
			cnf_values = fillCnfValues(cnf_matrix)
			currentTPR = cnf_values[3][4]
			currentAccuracy = cnf_values[3][8]
			TPRMatrix[i][x-trainingData_start] = currentTPR
			accuracyMatrix[i][x-trainingData_start] = currentAccuracy
			if max_TPR_classifiers[i]<currentTPR:
				max_TPR_classifiers[i] = currentTPR
				trainingDataForMaxTPR[i] = tempTrainingVectors
				maxTPR_Z[i] = Z
				maxTPR_test_y[i] = test_y
			if max_accuracy_classifiers[i]<currentAccuracy:
				max_accuracy_classifiers[i] = currentAccuracy
				trainingDataForMaxAccuracy[i] = tempTrainingVectors
				maxAcc_Z[i] = Z
				maxAcc_test_y[i] = test_y

def measure1():
	"""
	Measure 1: Print maximum TPR & accuracy of the model over the given training size
	defined by trainingData_start and trainingData_end
	"""
	print "Maximum TPR(from 0 to 1) of the 5 classifiers is: ", [round(x, 3) for x in max_TPR_classifiers]
	print "Maximum TPR is attained for training size: ", trainingDataForMaxTPR, "\n"
	print "Maximum Accuracy(from 0 to 1) of the 5 classifiers is: ", [round(x, 3) for x in max_accuracy_classifiers]
	print "Maximum Accuracy is attained for training size: ", trainingDataForMaxAccuracy, "\n"

def measure2():
	"""
	Plot TPR vs training size.
	Plot Accuracy vs training size.
	"""
	# Scatter plot the maximum (max_TPR, training size) points of each classifier
	for x in xrange(0, 5):
		plt.plot(range(trainingData_start, trainingData_end), TPRMatrix[x], label=labels[x])
		# Plot the maximum TPR points
	plt.scatter(trainingDataForMaxTPR, max_TPR_classifiers)
		# Annotate the maximum TPR points
	for i in xrange(0, 5):
		pt_label = "(" + str(trainingDataForMaxTPR[i]) + "," + "{0:.3f}".format(max_TPR_classifiers[i]) + ")"
		plt.annotate(pt_label, xy=(trainingDataForMaxTPR[i], max_TPR_classifiers[i]),
									xytext=(4,4), textcoords='offset points', fontsize='x-small')
	#TODO: Point annotations are overlapping. Correct it.
	legend = plt.legend(loc = 'upper left', shadow=True, fontsize=8)
	legend.get_frame()
	plt.title("Graph 1: TPR vs Training size")
	plt.xlabel("Training size")
	plt.ylabel("TPR(0.0-1.0)")

	# Scatter plot the maximum (max_accuracy, training size) points of each classifier
	plt.figure()
	for x in xrange(0, 5):
		plt.plot(range(trainingData_start, trainingData_end), accuracyMatrix[x], label=labels[x])
		# Plot the maximum TPR points
	plt.scatter(trainingDataForMaxAccuracy, max_accuracy_classifiers)
		# Annotate the maximum TPR points
	for i in xrange(0, 5):
		pt_label = "(" + str(trainingDataForMaxAccuracy[i]) + "," + "{0:.3f}".format(max_accuracy_classifiers[i]) + ")"
		plt.annotate(pt_label, xy=(trainingDataForMaxAccuracy[i], max_accuracy_classifiers[i]),
									xytext=(4,4), textcoords='offset points', fontsize='x-small')
	legend = plt.legend(loc = 'upper left', shadow=True, fontsize=8)
	legend.get_frame()
	plt.title("Graph 2: Accuracy vs Training size")
	plt.xlabel("Training size")
	plt.ylabel("Accuracy(0.0-1.0)")


def measure3():
	"""
	Plot the confusion matrix for each classifier when the TPR is the most.
	"""
	plt.figure(figsize = (11,5))
	np.set_printoptions(precision=3)
	for i in xrange(0,5):
		print "Analysis of classifier", labels[i]
		print "================================="
		cnf_matrix = confusion_matrix(maxTPR_test_y[i], maxTPR_Z[i])
		# Plot non-normalized confusion matrix
		plt.subplot(2,5,i+1)
		plot_confusion_matrix(cnf_matrix, classes=class_names,
													title=labels[i])
		# Plot normalized confusion matrix
		plt.subplot(2,5,i+6)
		plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
													title=labels[i])
		# print "#######", len(cnf_matrix[0]), cnf_matrix, maxTPR_test_y, maxTPR_Z
		showCnfValues(cnf_matrix)
		print "\n"
	plt.show()


# NOTE: Change the value of noOfTotalClasses, noOfTrainingVectors
# and noOfTestingVectors in actual use.
#: Total number of classes.
noOfTotalClasses = 3
#: Total number of vectors available for one class.
noOfTotalVectors = 400
#: For training purposes for one class use first `noOfTrainingVectors` vectors.
noOfTrainingVectors = 350
#: For testing purposes for one class use first `noOfTestingVectors` vectors.
noOfTestingVectors = noOfTotalVectors - noOfTrainingVectors
#: Each vector contains `noOfFeatures` features.
noOfFeatures = 31
#: This contains the no of classifiers defined below
noOfClassifiers = 5
#: This contains the path for the dataset.
datasetPath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
datasetPath = datasetPath + os.sep + "DSL-StrongPasswordData.csv"
max_training = 0
max_scores = []
trainingData_start = 2
trainingData_end = trainingData_start + noOfTrainingVectors

# Metrics for saving state of classifier at maximum TPR & Accuracy
#: Maximum TPR of each classifier
max_TPR_classifiers = [0]*5
#: Maximum accuracy of each classifier
max_accuracy_classifiers = [0]*5
#: Amount of training data required to achieve maximum TPR
trainingDataForMaxTPR = [0]*5
#: Amount of training data required to achieve maximum accuracy
trainingDataForMaxAccuracy = [0]*5
#: Each row represents a classifier's values in TPRMatrix.
#: These values are the TPR values of the classifier for each possible training size(column).
TPRMatrix = [[0 for x in range(noOfTrainingVectors)] for x in range(5)]
#: Each row represents a classifier's values in AccuracyMatrix.
#: These values are the accuracy values of the classifier for each possible training size(column).
accuracyMatrix = [[0 for x in range(noOfTrainingVectors)] for x in range(5)]

#: maxTPR_Z contains the predicted values when the TPR is maximum
maxTPR_Z = [[0 for x in range(noOfTestingVectors)] for x in range(5)]
#: maxAcc_Z contains the predicted values when the accuracy is maximum
maxAcc_Z = [[0 for x in range(noOfTestingVectors)] for x in range(5)]
#: maxTPR_test_y contains the test targets values when the TPR is maximum
maxTPR_test_y = [[0 for x in range(noOfTestingVectors)] for x in range(5)]
#: maxAcc_test_y contains the test targets when the accuracy is maximum
maxAcc_test_y = [[0 for x in range(noOfTestingVectors)] for x in range(5)]

class_names = getClassNames()
labels = ['SVC(linear)', 'LinearSVC', 'SVC(rbf)', 'SVC(poly)', 'NuSVC(rbf)']
#: These are the column names of the matrix filled by fillCnfValues()
columnNames = ["TP", "FN", "FP", "TN", "TPR", "TNR", "FNR", "FPR", "Accuracy"]


# C is SVM Regularization parameter
trainAndTest(C=1.0)

# ==========================
# MEASUREMENT OF PERFORMANCE
# ==========================
# Measure 1: Print maximum TPR & accuracy of the model.
measure1()
# Measure 2: Plot the TPR vs training size and Accuracy vs training size.
measure2()
# Measure 3: Plot the confusion matrix for each classifier when the TPR is the most.
measure3()
# TODO: Measure 4: Plot the performance of each classifier for higher no. of classes when training size is set at the point of maximum TPR for 3 classes.
