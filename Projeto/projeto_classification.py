import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import CenterCrop,Rescaling

#Bayes classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#KNN Classification
from sklearn import neighbors, datasets
from sklearn.inspection import DecisionBoundaryDisplay


X = np.load('numpy_files/Xtrain_Classification1.npy')
y = np.load('numpy_files/ytrain_Classification1.npy')
#print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)
print(X.shape)
X_alt = X.reshape((8273,30, 30,3))
Y_alt = y
print(X_alt.shape)

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
#dense = keras.layers.Dense(units=16)
inputs = keras.Input(shape=(X_train.shape))


x = CenterCrop(height=30, width=30)(inputs)
# Rescale images to [0, 1]
x = Rescaling(scale=1.0 / 255)(x)

# Apply some convolution and pooling layers
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)

# Apply global average pooling to get flat feature vectors
#x = layers.GlobalAveragePooling2D()(x)

# Add a dense classifier on top
num_classes = 10
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)


model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X_train, y_train, batch_size=1, epochs=1)

#predictions = model.predict(X_test)
#print(predictions.shape)