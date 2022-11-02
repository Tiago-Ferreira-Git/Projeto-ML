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
from sklearn.metrics import f1_score, confusion_matrix,balanced_accuracy_score
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
X = np.load('numpy_files/classification/Xtrain_Classification2.npy')
y = np.load('numpy_files/classification/Ytrain_Classification2.npy')
print(X.shape,y.shape)
print("Class 0 points : %d" % (y == 0).sum())
print("Class 1 points : %d" % (y == 1).sum())
print("Class 2 points : %d" % (y == 2).sum())
def balance_dataset(X,y):
    print(X.shape)
    class0 = (y == 0).sum()
    class1 = (y == 1).sum()
    class2 = (y == 2).sum()
    to_be_alt = np.where(np.logical_or(y == 2 , y == 0))
    to_be_alt = to_be_alt[0]
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ]
    )
    X_alt = (X.reshape((X.shape[0],5,5,3)))
    Y_alt = y
    difference  = int((class1-class0-class2))
    print(difference)
    for i in range(difference):
        index = random.randint(0, to_be_alt.shape[0]-1)
        
        augmented_images = data_augmentation(X_alt[to_be_alt[index]])
        augmented_images = augmented_images.numpy().astype("uint64")
        augmented_images = (augmented_images.reshape((1,75)))
        X_alt = (X_alt.reshape((X_alt.shape[0],75)))
        X_alt = np.append(X_alt,augmented_images,axis=0)
        X_alt = (X_alt.reshape((X_alt.shape[0],5, 5,3)))
        Y_alt = np.append(Y_alt,Y_alt[to_be_alt[index]])
        print(X_alt.shape,Y_alt.shape)
    X_alt = (X_alt.reshape((X_alt.shape[0],75)))
    np.save("numpy_files/Xtrain_Classification2_altered", X_alt)
    np.save("numpy_files/Ytrain_Classification2_altered", Y_alt)
    return X_alt,Y_alt


#X,y = balance_dataset(X,y)
#X = np.load('numpy_files/Xtrain_Classification2_altered_90.npy')
#y = np.load('numpy_files/Ytrain_Classification2_altered_90.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# #Bayes Classification Gaussian
# gnb = GaussianNB()
# y_pred = gnb.fit(X_train, y_train).predict(X_test)
# print("Gaussian - Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != y_pred).sum()))
# print(f"Balanced score: {balanced_accuracy_score(y_test, y_pred)}")
# gnb_f1 = balanced_accuracy_score(y_test, y_pred)

# #Bayes Classification Multinomial
# mnb = MultinomialNB()
# y_pred = mnb.fit(X_train, y_train).predict(X_test)
# print("Multinomial - Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != y_pred).sum()))
# print(f"Balanced score: {balanced_accuracy_score(y_test, y_pred)}")
# mnb_f1 = f1_score(y_test, y_pred, average='macro')

# #Bayes Classification Complement
# cnb = ComplementNB()
# y_pred = cnb.fit(X_train, y_train).predict(X_test)
# print("Complement - Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != y_pred).sum()))
# print(f"Balanced score: {balanced_accuracy_score(y_test, y_pred)}")
# cnb_f1 = f1_score(y_test, y_pred, average='macro')



# #Bayes Classification Bernoulli
# bnb = BernoulliNB()
# y_pred = bnb.fit(X_train, y_train).predict(X_test)
# print("Bernoulli - Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != y_pred).sum()))
# print(f"Balanced score: {balanced_accuracy_score(y_test, y_pred)}")
# bnb_f1 = f1_score(y_test, y_pred, average='macro')


# #Bayes Classification Categorical - it assumes that X is encoded
# cnb = CategoricalNB()
# y_pred = cnb.fit(X_train, y_train).predict(X_test)
# print("Categorical - Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0], (y_test != y_pred).sum()))
# print(f"F1 score: {f1_score(y_test, y_pred, average='binary')}")




# #Knn Classification
# n_neighbors = 20
# #weights = ["uniform", "distance"]
# weights = "uniform"
# clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("KNN - Number of mislabeled points out of a total %d points : %d"
#       % (X_test.shape[0], (y_test != y_pred).sum()))
# print(f"Balanced score: {balanced_accuracy_score(y_test, y_pred)}")
# knn_f1 = f1_score(y_test, y_pred, average='macro')

# #SVM kernel rfb

# clf = svm.SVC()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("SVM (rfb) - Number of mislabeled points out of a total %d points : %d"
#       % (X_test.shape[0], (y_test != y_pred).sum()))
# print(f"Balanced score: {balanced_accuracy_score(y_test, y_pred)}")
# svm_rfb = f1_score(y_test, y_pred, average='macro')




# #SVM kernel sigmoid
# clf = svm.SVC(kernel="sigmoid")
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("SVM  (sigmoid) - Number of mislabeled points out of a total %d points : %d"
#       % (X_test.shape[0], (y_test != y_pred).sum()))
# print(f"Balanced score: {balanced_accuracy_score(y_test, y_pred)}")
# svm_sigm = f1_score(y_test, y_pred, average='macro')


# #SVM kernel poly
# clf = svm.SVC(kernel="poly")
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("SVM  (poly) - Number of mislabeled points out of a total %d points : %d"
#       % (X_test.shape[0], (y_test != y_pred).sum()))
# print(f"Balanced score: {balanced_accuracy_score(y_test, y_pred)}")
# svm_poly = f1_score(y_test, y_pred, average='macro')




# svm = {"svm_rfb": svm_rfb,"svm_sigm": svm_sigm,"svm_poly":svm_poly, "label": "experiment"}
# savemat("mat_files/svm.mat",svm)

#CNN

#keras image classification

X_train = (X_train.reshape((X_train.shape[0],5, 5,3)))
X_test = (X_test.reshape((X_test.shape[0],5, 5,3)))
ohe.fit(y_train.reshape(y_train.shape[0],1))
y_train = ohe.transform(y_train.reshape(y_train.shape[0],1)).toarray()


inputs = keras.Input(shape=(5,5,3))

x = CenterCrop(height=5, width=5)(inputs)
x = Rescaling(scale=1.0 / 255)(inputs)


x = layers.Conv2D(filters=64, kernel_size=(3, 3),input_shape=(5,5,3), activation="relu",strides=1,padding="same")(x)

x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu",padding="same")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu",padding="same")(x)


x=layers.Flatten()(x)
x=layers.Dense(300,activation="relu")(x)
x=layers.Dense(300,activation="relu")(x)



# Add a dense classifier on top
num_classes = 3
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=64, epochs=20)
#model.summary()

y_pred = model.predict(X_test)
y_pred = ohe.inverse_transform(y_pred)
y_test = y_test.reshape(1,y_test.shape[0])
y_pred = y_pred.reshape(1,y_pred.shape[0])
y_test = (np.array(y_test)[0])
y_pred = (np.array(y_pred)[0])

#print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print(f"Balanced score: {balanced_accuracy_score(y_test, y_pred)}")
cnn = f1_score(y_test,y_pred, average='macro')

#f1_final = {"svm_rfb": svm_rfb,"svm_sigm": svm_sigm,"svm_poly":svm_poly,"cnn":cnn, "label": "experiment"}
#savemat("mat_files/f1_final.mat",f1_final)
#confusion matrix para o que escolhermos
#confusion_matrix(y_true, y_pred)