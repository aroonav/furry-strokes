import os
import numpy as np
import csv
from sklearn.neural_network import MLPClassifier
import pyxhook.pyxhook as pyxhook
import time
import csv
import sys
from sklearn import svm
from termios import tcflush, TCIOFLUSH

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
datasetPath = datasetPath + os.sep + "OURdata.csv"


noOfInputNodes = 31
# The number of Hidden nodes is taken as (2*P)/3, where P is the number of the input nodes
noOfHiddenNodes = 15
# The number of output nodes is equal to the number of classes
noOfOutputNodes = noOfTotalClasses

# X: We take all the features. Or we can take only some features here by slicing.
# y: This contains the actual classes for each training vector i.e the target.
X,y = load_trainingData()

# Neural network classifiers
sgd_clf = MLPClassifier(hidden_layer_sizes = (noOfInputNodes, noOfHiddenNodes, noOfOutputNodes), 
		activation = "tanh", solver = "sgd", max_iter = 1200, learning_rate = "adaptive")
adam_clf = MLPClassifier(hidden_layer_sizes = (noOfInputNodes, noOfHiddenNodes, noOfOutputNodes), 
		activation = "tanh", solver = "adam", max_iter = 1000)
lbfgs_clf = MLPClassifier(hidden_layer_sizes = (noOfInputNodes, noOfHiddenNodes, noOfOutputNodes), 
		activation = "tanh", solver = "lbfgs", max_iter = 1000)
# SVM Classifiers
C = 1.0  # SVM regularization parameter
svc_clf = svm.SVC(kernel='linear', C=C)
lin_svc_clf = svm.LinearSVC(C=C)
rbf_svc_clf = svm.SVC(kernel='rbf', gamma=0.7, C=C)
nu_svc_clf = svm.NuSVC()

# Training of neural networks classifiers
sgd_clf.fit(X,y)
adam_clf.fit(X,y)
lbfgs_clf.fit(X,y)
# Training of SVM classifiers
svc = svc_clf.fit(X, y)
lin_svc = lin_svc_clf.fit(X, y)
rbf_svc = rbf_svc_clf.fit(X, y)
nu_svc = nu_svc_clf.fit(X, y)



print "Model Trained"
print "\nThe password to type is: .xat17padn"
print "\nEnter the password and we will predict the rest:"


############################################################


class KeyStroke:
	def __init__(self, key, mname, time):
		self.key = key
		self.mname = mname
		self.time = time

	
class StrokesLine:
	def __init__(self, UpArray, DownArray):
		self.UpArray = UpArray
		self.DownArray = DownArray
		self.line = ""

	def returnVector(self):
		line = []
		for i in range(len(self.UpArray) - 1):
			holdTime = float(self.UpArray[i].time) - float(self.DownArray[i].time)
			downDownTime = float(self.DownArray[i+1].time) - float(self.DownArray[i].time)
			upDownTime = float(self.DownArray[i+1].time) - float(self.UpArray[i].time)
			line.append(str(holdTime))
			line.append(str(downDownTime))
			line.append(str(upDownTime))
		line.append(str(self.UpArray[i+1].time - self.DownArray[i+1].time))

		return line

# This function is called at each iteration to check if the password entered is correct
def checkPass(array):
	global authPass
	typedPass = ""
	for i in range(len(array)):
		typedPass += array[i].key
	# print("Typed Password: "+typedPass+"\n      Password: "+authPass)
	if authPass == typedPass:
		return True
	else:
		return False
		
# This function is called every time a key is presssed
def kbevent(event):
    global running
    global UpArray, DownArray
    # print key info
    # tm = repr(time.time())
    tm = time.time()

    keyEvent = KeyStroke(event.Key, event.MessageName, tm)
    if keyEvent.mname == "key down":
    	DownArray.append(keyEvent)
    else:
    	UpArray.append(keyEvent)

    # If the ascii value matches carriage return, terminate the while loop
    if event.Ascii == 13:
    	if keyEvent.mname == "key up":
    		running = False


UpArray = []
DownArray = []
# Create hookmanager
hookman = pyxhook.HookManager()
# Define our callback to fire when a key is pressed down
hookman.KeyDown = kbevent
hookman.KeyUp = kbevent
# Hook the keyboard
hookman.HookKeyboard()
# Start our listener

hookman.start()
# Create a loop to keep the application running
running = True
while running:
    time.sleep(0.00001)

# Close the listener when we are done
hookman.cancel()
tcflush(sys.stdin, TCIOFLUSH)

writer = StrokesLine(UpArray, DownArray)
testing_vector = writer.returnVector()
testing_vector = map(float, testing_vector)
testing_vector = [[testing_vector[x] for x in range(len(testing_vector))]]

print "\nOkay, testing against our classifiers..."

classifiers = ['NN(SGD)', 'NN(Adam)', 'NN(Lbfgs)', 'SVC(linear)', 'LinearSVC', 'SVC(rbf)', 'SVC(poly)', 'NuSVC(rbf)']
for i, clf in enumerate((sgd_clf, adam_clf, lbfgs_clf, svc, lin_svc, rbf_svc, nu_svc)):
	# Pass testing data to the classifier
	Z = clf.predict(testing_vector)
	print "\n", classifiers[i],"predicts:", Z[0]
