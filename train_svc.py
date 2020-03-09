# USAGE
# python train.py

# import the necessary packages
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from pyimagesearch import config
import numpy as np
import pickle
import os

def load_data_split(splitPath):
	# initialize the data and labels
	data = []
	labels = []

	# loop over the rows in the data split file
	for row in open(splitPath):
		# extract the class label and features from the row
		row = row.strip().split(",")
		label = row[0]
		features = np.array(row[1:], dtype="float")

		# update the data and label lists
		data.append(features)
		labels.append(label)

	# convert the data and labels to NumPy arrays
	data = np.array(data)
	labels = np.array(labels)

	# return a tuple of the data and labels
	return (data, labels)

# derive the paths to the training and testing CSV files
trainingPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.TRAIN)])
testingPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.TEST)])

# load the data from disk
print("[INFO] loading data...")
(trainX, trainY) = load_data_split(trainingPath)
(testX, testY) = load_data_split(testingPath)

# load the label encoder from disk
le = pickle.loads(open(config.LE_PATH, "rb").read())


# train the model SVC looking for the best value for hyperparameters "C"
print("[INFO] train the model SVC ")
tuning_param = [{
          "C": [0.01, 0.1, 1, 10, 100]
         }]
 
svm = LinearSVC(penalty='l2', loss='squared_hinge')  # As in Tang (2013)
clf = GridSearchCV(svm, tuning_param, cv=10)
clf.fit(trainX, trainY)
print("Best value of 'C':" , clf.best_params_)

# evaluate the model SVC 
print("[INFO] evaluating model SVC ...")
preds = clf.predict(testX)
preds_train = clf.predict(trainX)
print(classification_report(testY, preds, target_names=le.classes_))
print("Testing dataset Accurarcy:", accuracy_score(testY, preds))
print("Training dataset Accuracy:", accuracy_score(trainY, preds_train))

# serialize the model to disk
print("[INFO] saving model...")
f = open("svc.cpickle", "wb")
f.write(pickle.dumps(clf))
f.close()
