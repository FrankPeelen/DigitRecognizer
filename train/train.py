import pandas
import numpy as np
from sklearn.externals import joblib
from helper import learningCurve
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

print("Loading Data")
data = pandas.read_csv("data/train.csv")
print("Loaded Data")

X = data.iloc[:,0:data.shape[1] - 1]
y = data["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

print("Training Algorithm")
alg = joblib.load('alg.pkl')
#alg = LogisticRegression(max_iter=1000, multi_class='ovr', solver='lbfgs', verbose=1)
#alg.fit(X_train, y_train)
print("Trained Algorithm")

print("iters = " + str(alg.n_iter_))

#learningCurve(alg, X, y)

print("Predicting on CV Set")
preds = alg.predict(X_val)
print("Predicted on CV Set")

results = preds == y_val
hits = 0
for result in results:
	if result:
		hits += 1

accuracy = hits / len(y_val)

print("Accuracy on Train Set: " + str(alg.score(X_train, y_train)))
print("Accuracy on CV Set: " + str(accuracy))

joblib.dump(alg, 'alg.pkl')
