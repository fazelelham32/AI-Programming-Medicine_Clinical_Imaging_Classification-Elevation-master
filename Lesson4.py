import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing

data = pd.read_csv("F:/NEW/importent folder/main folder/projects for github/AI and AI in Python/codes and files/car.data")
print(data.head())

le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
print(buying)
