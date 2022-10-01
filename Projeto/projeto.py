from cv2 import mean
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

plt.close('all')
X = np.load('Xtrain_Regression1.npy')
Y = np.load('Ytrain_Regression1.npy')


def linear_regression(X,Y):
    X2 = np.empty(X.shape)
    for i,column in enumerate(X.T):
        X2.T[i] = X.T[i] - np.mean(column)
    Y2 = Y - np.mean(Y)
    beta_aux = X2.transpose().dot(X2)

    beta_aux = np.linalg.inv( beta_aux )

    beta = (beta_aux.dot(X2.transpose())).dot(Y2)
    print(f"Sum of squared errors {mean_squared_error(Y2, X2.dot(beta))} my linear regression")


lasso_vector = np.arange(0.1,2, 0.1)
mean_squared_error_x = np.empty(lasso_vector.shape)
for i,alpha in enumerate(lasso_vector):
    reg = linear_model.Lasso(alpha)
    reg.fit(X, Y)
    Y_pred = reg.predict(X)
    mean_squared_error_x[i] = mean_squared_error(Y, Y_pred)

regr = linear_model.LinearRegression()
regr.fit(X, Y)
linear_regression(X,Y)
Y_pred = regr.predict(X)
print(f"Sum of squared errors {mean_squared_error(Y, Y_pred)} sklearn regression")

plt.figure(1) 
plt.xlabel("Lasso lambda") 
plt.ylabel("Mean squared error")
plt.plot(lasso_vector,mean_squared_error_x)
plt.show()

"""
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption")
plt.figure(1) 
plt.plot(X,Y,"c.")
plt.plot(X,X.dot(beta),"b.")
plt.figure(2) 
plt.plot(X2,Y2,"c.")
plt.plot(X2,X2.dot(beta),"b.")
plt.show()
plt.pause(1)
plt.close('all')"""


