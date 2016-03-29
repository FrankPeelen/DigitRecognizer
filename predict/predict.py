import pandas
import numpy as np
import math
import time

print("Loading Data")
X = pandas.read_csv("data/test.csv")
print("Loaded Data")

from sklearn.externals import joblib

alg = joblib.load('alg.pkl')

print("Predicting on test Set")
preds = alg.predict(X)
print("Predicted on test Set")

pandas.DataFrame({
    	"ImageId": range(1,len(X) + 1),
      	"Label": preds
	}).to_csv('submission' + str(math.floor(time.time())) + '.csv',index=False)
