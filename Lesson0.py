import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection as ms
from sklearn import linear_model

def linear_score(x, y):
    # Here, you need to calculate or provide the score based on inputs x and y
    # Replace the placeholder 'score' with the appropriate calculation or value
    score = 0.85  # Replace with your own score calculation

    return score

data = pd.read_csv('F:/NEW/importent folder/main folder/projects for github/AI and AI in Python/codes and files/student-mat.csv', sep=';')
print(data.head())

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
predict = "G3"
# Assuming you have your data defined here

x = np.array(data.drop(labels=[predict], axis=1))
y = np.array(data[predict])

for _ in range(10):
    x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear_score(x_test, y_test)
    print(acc)