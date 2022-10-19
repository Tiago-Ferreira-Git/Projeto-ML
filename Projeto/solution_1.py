import numpy as np
from sklearn import linear_model


X_train = np.load('Xtrain_Regression1.npy')
Y_train = np.load('Ytrain_Regression1.npy')
X_test = np.load('Xtest_Regression1.npy')


reg = linear_model.Lasso(0.017)
reg.fit(X_train, Y_train)
Y_test = reg.predict(X_test)

np.save("Ytest_Regression1", Y_test)