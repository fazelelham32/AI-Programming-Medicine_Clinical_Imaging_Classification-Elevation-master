import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection as ms
from sklearn import linear_model

import pickle

data = pd.read_csv('F:/NEW/importent folder/main folder/projects for github/AI and AI in Python/codes and files/student-mat.csv', sep=';')
print(data.head())

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
predict = "G3"
# Assuming you have your data defined here

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#best=0

#for _ in range(100):


#    
 #   linear = linear_model.LinearRegression()
 #   linear.fit(x_train, y_train)
 #   acc = linear.score(x_test, y_test)
 #   print(acc)

#if (acc > best):
 #   best=acc
 #   with open("studentmodel.pickle", "wb") as f:
  #      pickle.dump(linear, f)


newModel = pickle.load(open("studentmodel.pickle", "rb"))

print("coefficient:", newModel.coef_)
print("Intercept:", newModel.intercept_)







