import pandas
import numpy as np

print("Loading Data")
data = pandas.read_csv("data/train.csv")
print("Loaded Data")

X = data.iloc[:,0:data.shape[1] - 1]
y = data["label"]

from sklearn.cross_validation import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

alg = OneVsRestClassifier(LogisticRegression(max_iter=10))
print("Training Algorithm")
alg.fit(X_train, y_train)
print("Trained Algorithm")

print("Predicting on CV Set")
preds = alg.predict(X_val)
print("Predicted on CV Set")

results = preds == y_val
hits = 0
for result in results:
	if result:
		hits += 1

accuracy = hits / len(y_val)

print("Accuracy on CV Set: " + str(accuracy))

from sklearn.externals import joblib
joblib.dump(alg, 'alg.pkl')
