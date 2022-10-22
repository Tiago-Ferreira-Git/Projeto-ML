import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#Bayes classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#KNN Classification
from sklearn import neighbors, datasets
from sklearn.inspection import DecisionBoundaryDisplay


X = np.load('numpy_files/Xtrain_Classification1.npy')
y = np.load('numpy_files/ytrain_Classification1.npy')
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)
#X = X.reshape((8273,30, 30,3))

#Bayes Classification
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))


#Knn Classification
n_neighbors = 20
#weights = ["uniform", "distance"]
weights = "uniform"
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))


#keras image classification