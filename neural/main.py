import os
import numpy as np
# import matplotlib.pyplot as plt
import csv
from sklearn.neural_network import MLPClassifier

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
noOfTotalClasses = 3
# Total number of vectors available for one class.
noOfTotalVectors = 400
# For training purposes for one class use first `noOfTrainingVectors` vectors.
noOfTrainingVectors = 300
# For testing purposes for one class use first `noOfTestingVectors` vectors.
# noOfTestingVectors = noOfTotalVectors - noOfTrainingVectors
noOfTestingVectors = 100
# Each vector contains `noOfFeatures` features.
noOfFeatures = 31
# This contains the path for the dataset.
datasetPath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
# datasetPath = datasetPath + os.sep + "DSL-StrongPasswordData.csv"
datasetPath = datasetPath + os.sep + "OURdata.csv"


noOfInputNodes = noOfFeatures
# The number of Hidden nodes is taken as (2*P)/3, where P is the number of the input nodes
noOfHiddenNodes = 15
# The number of output nodes is equal to the number of classes
noOfOutputNodes = noOfTotalClasses

# X: We take all the features. Or we can take only some features here by slicing.
# y: This contains the actual classes for each training vector i.e the target.
X,y = load_trainingData()
test_X,test_y = load_testingData()

sgd_clf = MLPClassifier(hidden_layer_sizes = (noOfInputNodes, noOfHiddenNodes, noOfOutputNodes), 
		activation = "tanh", solver = "sgd", max_iter = 1200, learning_rate = "adaptive")
adam_clf = MLPClassifier(hidden_layer_sizes = (noOfInputNodes, noOfHiddenNodes, noOfOutputNodes), 
		activation = "tanh", solver = "adam", max_iter = 1000)
lbfgs_clf = MLPClassifier(hidden_layer_sizes = (noOfInputNodes, noOfHiddenNodes, noOfOutputNodes), 
		activation = "tanh", solver = "lbfgs", max_iter = 1000)

sgd_clf.fit(X,y)
adam_clf.fit(X,y)
lbfgs_clf.fit(X,y)

for i, clf in enumerate((sgd_clf, adam_clf, lbfgs_clf)):
	if(i==0):
		print "Testing with SGD solver."
	elif(i==1):
		print "Testing with Adam solver"
	else:
		print "Testing with Lbfgs solver."	
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