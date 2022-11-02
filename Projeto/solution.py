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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import CenterCrop,Rescaling
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
X_train = np.load('Xtrain_Classification1.npy')
X_test = np.load('Xtest_Classification1.npy')
Y_train = np.load('ytrain_Classification1.npy')
print(X_test.shape)
def balance_binary_dataset(X,y):
    
    to_be_alt = np.where(y == 1)
    to_be_alt = to_be_alt[0]
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ]
    )
    X_alt = (X.reshape((X.shape[0],30, 30,3)))
    Y_alt = y
    for i in range((y == 0).sum()-(y == 1).sum()):
        index = random.randint(0, to_be_alt.shape[0]-1)
        
        augmented_images = data_augmentation(X_alt[to_be_alt[index]])
        augmented_images = augmented_images.numpy().astype("uint64")
        augmented_images = (augmented_images.reshape((1,2700)))
        X_alt = (X_alt.reshape((X_alt.shape[0],2700)))
        X_alt = np.append(X_alt,augmented_images,axis=0)
        X_alt = (X_alt.reshape((X_alt.shape[0],30, 30,3)))
        Y_alt = np.append(Y_alt,Y_alt[to_be_alt[index]])
    X_alt = (X_alt.reshape((X_alt.shape[0],2700)))
    return X_alt,Y_alt


X_train,Y_train = balance_binary_dataset(X_train,Y_train)
print("Balanced ")


X_train = (X_train.reshape((X_train.shape[0],30, 30,3)))
X_test = (X_test.reshape((X_test.shape[0],30, 30,3)))
ohe.fit(Y_train.reshape(Y_train.shape[0],1))
Y_train = ohe.transform(Y_train.reshape(Y_train.shape[0],1)).toarray()


inputs = keras.Input(shape=(30,30,3))

x = CenterCrop(height=30, width=30)(inputs)
x = Rescaling(scale=1.0 / 255)(x)


x = layers.Conv2D(filters=64, kernel_size=(3, 3),input_shape=(30,30,3), activation="relu",strides=1,padding="same")(x)

x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu",padding="same")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu",padding="same")(x)


x=layers.Flatten()(x)
x=layers.Dense(300,activation="relu")(x)
x=layers.Dense(300,activation="relu")(x)



# Add a dense classifier on top
num_classes = 2
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss='binary_crossentropy',metrics=["accuracy"])
model.fit(X_train, Y_train, batch_size=32, epochs=10)
model.summary()

Y_test = model.predict(X_test)
Y_test = ohe.inverse_transform(Y_test)
Y_test = Y_test.reshape(1,Y_test.shape[0])
Y_test = (np.array(Y_test)[0])
print(Y_test.shape)
np.save("Ytest_Classification1", Y_test)