from scipy.io import savemat
import random
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
from sklearn.metrics import f1_score, confusion_matrix
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
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
ohe = OneHotEncoder()
X = np.load('numpy_files/Xtrain_Classification1_altered.npy')
y = np.load('numpy_files/ytrain_Classification1.npy')
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(y.shape)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = CenterCrop(height=30, width=30)(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for size in [8]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)


    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    #x = layers.GlobalAveragePooling2D()(x)    
    activation = "softmax"
    units = 2

    x=layers.Flatten()(x)
    x=layers.Dense(20,activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x=layers.Dense(20,activation="relu")(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


X_train = (X_train.reshape((X_train.shape[0],30, 30,3)))
X_test = (X_test.reshape((X_test.shape[0],30, 30,3)))
ohe.fit(y_train.reshape(y_train.shape[0],1))
y_train = ohe.transform(y_train.reshape(y_train.shape[0],1)).toarray()

model = make_model(input_shape=(30,30,3), num_classes=2)
model.compile(optimizer="adam", loss='binary_crossentropy',metrics=["accuracy"])
model.summary()
model.fit(X_train, y_train, batch_size=32, epochs=20)


y_pred = model.predict(X_test)
y_pred = ohe.inverse_transform(y_pred)
y_test = y_test.reshape(1,y_test.shape[0])
y_pred = y_pred.reshape(1,y_pred.shape[0])
y_test = (np.array(y_test)[0])
y_pred = (np.array(y_pred)[0])

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print(f"F1 score: {f1_score(y_test,y_pred, average='binary')}")