from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut,cross_val_score
cv = LeaveOneOut()
plt.close('all')
X = np.load('Xtrain_Regression1.npy')
Y = np.load('Ytrain_Regression1.npy')


#dc - data centered
def linear_regression_dc(X,Y):
    X2 = np.empty(X.shape)
    for i,column in enumerate(X.T):
        X2.T[i] = X.T[i] - np.mean(column)
    Y2 = Y - np.mean(Y)
    beta_aux = X2.transpose().dot(X2)

    beta_aux = np.linalg.inv( beta_aux )

    beta = (beta_aux.dot(X2.transpose())).dot(Y2)
    print(f"Sum of squared errors {mean_squared_error(Y2, X2.dot(beta))} my centered linear regression")

def linear_regression(X,Y):
    n,m=np.shape(X)
    Xn=np.hstack((np.ones((n,1)),X))

    beta = ((np.linalg.inv((Xn.T).dot(Xn))).dot(Xn.T)).dot(Y)
    print(f"Sum of squared errors {mean_squared_error(Y, Xn.dot(beta))} my linear regression with beta0")


lasso_vector = np.arange(0.005,0.02, 0.001)
mean_squared_error_x = np.empty(lasso_vector.shape)
betas = np.arange(10.0*lasso_vector.shape[0])
betas = betas.reshape((10, lasso_vector.shape[0]))
betas = np.zeros_like(betas)
for i,alpha in enumerate(lasso_vector):
    reg = linear_model.Lasso(alpha)
    reg.fit(X, Y)
    Y_pred = reg.predict(X)
    beta = reg.coef_
    for j,beta_i in enumerate(beta):
        betas[j][i] = beta_i
   
    mean_squared_error_x[i] = mean_squared_error(Y, Y_pred)
    scores = cross_val_score(reg, X, Y, scoring='neg_mean_absolute_error',cv=cv, n_jobs=-1)
    #print(f"LOO score: {np.mean(np.absolute(scores))} | MSE: {mean_squared_error_x[i]} for alpha = {alpha}")
    #print(beta)
    
    

regr = linear_model.LinearRegression()
regr.fit(X, Y)
print(regr.coef_)
Y_pred = regr.predict(X)
#print(f"Sum of squared errors {mean_squared_error(Y, Y_pred)} sklearn regression")


linear_regression(X,Y)


scores = cross_val_score(regr, X, Y, scoring='neg_mean_absolute_error',cv=cv, n_jobs=-1)
#print(f"LeaveOneOut score for linear regression {np.mean(np.absolute(scores))}")


to_plot = {"lasso_vector": lasso_vector,"mean_squared_error_x": mean_squared_error_x, "label": "experiment"}
savemat("to_plot_lasso_mse.mat",to_plot)
to_plot = {"lasso_vector": lasso_vector,"betas": betas, "label": "experiment"}
savemat("to_plot_lasso_betas.mat",to_plot)





