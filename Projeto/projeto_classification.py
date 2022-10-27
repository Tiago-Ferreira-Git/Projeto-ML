from scipy.io import savemat
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""


import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder

#Redes neuronais
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import CenterCrop,Rescaling

#Bayes classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB ,ComplementNB,BernoulliNB,CategoricalNB

#KNN Classification
from sklearn import neighbors, datasets
from sklearn.inspection import DecisionBoundaryDisplay

#SVM
from sklearn import svm


ohe = OneHotEncoder()
X = np.load('numpy_files/Xtrain_Classification1.npy')
y = np.load('numpy_files/ytrain_Classification1.npy')
#print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X.shape)

#Bayes Classification Gaussian
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Gaussian - Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != y_pred).sum()))
print(f"F1 score: {f1_score(y_test, y_pred, average='binary')}")


#Bayes Classification Multinomial
mnb = MultinomialNB()
y_pred = mnb.fit(X_train, y_train).predict(X_test)
print("Multinomial - Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != y_pred).sum()))
print(f"F1 score: {f1_score(y_test, y_pred, average='binary')}")


#Bayes Classification Complement
cnb = ComplementNB()
y_pred = cnb.fit(X_train, y_train).predict(X_test)
print("Complement - Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != y_pred).sum()))
print(f"F1 score: {f1_score(y_test, y_pred, average='binary')}")




#Bayes Classification Bernoulli
bnb = BernoulliNB()
y_pred = bnb.fit(X_train, y_train).predict(X_test)
print("Bernoulli - Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != y_pred).sum()))
print(f"F1 score: {f1_score(y_test, y_pred, average='binary')}")



# #Bayes Classification Categorical - it assumes that X is encoded
# cnb = CategoricalNB()
# y_pred = cnb.fit(X_train, y_train).predict(X_test)
# print("Categorical - Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != y_pred).sum()))
# print(f"F1 score: {f1_score(y_test, y_pred, average='binary')}")




#Knn Classification
n_neighbors = 20
#weights = ["uniform", "distance"]
weights = "uniform"
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("KNN - Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
print(f"F1 score: {f1_score(y_test, y_pred, average='binary')}")


#SVM

clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("SVM - Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
print(f"F1 score: {f1_score(y_test, y_pred, average='binary')}")




#CNN

#keras image classification

X_train = (X_train.reshape((X_train.shape[0],30, 30,3)))
X_test = (X_test.reshape((X_test.shape[0],30, 30,3)))
ohe.fit(y_train.reshape(y_train.shape[0],1))
y_train = ohe.transform(y_train.reshape(y_train.shape[0],1)).toarray()


inputs = keras.Input(shape=(30,30,3))

x = CenterCrop(height=30, width=30)(inputs)
# Rescale images to [0, 1]
x = Rescaling(scale=1.0 / 255)(x)

# Apply some convolution and pooling layers
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="softmax",padding="same")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="softmax",padding="same")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="softmax",padding="same")(x)

# Apply global average pooling to get flat feature vectors
x = layers.GlobalAveragePooling2D()(x)

            

# Add a dense classifier on top
num_classes = 2
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)


model = keras.Model(inputs=inputs, outputs=outputs)
#model.compile(optimizer="adam", loss='categorical_crossentropy')
model.compile(optimizer="sgd", loss='binary_crossentropy',metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=32, epochs=10)
#model.summary()

y_pred = model.predict(X_test)
y_pred = ohe.inverse_transform(y_pred)
y_test = y_test.reshape(1,y_test.shape[0])
y_pred = y_pred.reshape(1,y_pred.shape[0])
y_test = (np.array(y_test)[0])
y_pred = (np.array(y_pred)[0])
# print(y_test)
# to_cmp = {"y_test": y_test,"y_pred": y_pred, "label": "experiment"}
# savemat("mat_files/to_cmp.mat",to_cmp)
# print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

# print(f"F1 score: {f1_score(y_test,y_pred, average='binary')}")

