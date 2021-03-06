# USAGE
# python train.py

# import the necessary packages
from sklearn.linear_model import LogisticRegression
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

# train the model
print("[INFO] training model Logistic Regression...")
logit = LogisticRegression(solver="lbfgs", multi_class="auto")
logit.fit(trainX, trainY)

# evaluate the model
print("[INFO] evaluating...")
preds_train = logit.predict(trainX)
preds = logit.predict(testX)
print(classification_report(testY, preds, target_names=le.classes_))
print("Testing dataset Accuracy:", accuracy_score(testY, preds))
print("Training dataset Accuracy:", accuracy_score(trainY, preds_train))

# serialize the model to disk
print("[INFO] saving model...")
f = open("logit.cpickle", "wb")
f.write(pickle.dumps(logit))
f.close()
