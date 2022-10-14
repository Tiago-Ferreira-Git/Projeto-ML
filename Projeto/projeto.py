from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import LeaveOneOut,cross_val_score
cv = LeaveOneOut()
plt.close('all')
X = np.load('Xtrain_Regression1.npy')
Y = np.load('Ytrain_Regression1.npy')


#Lasso
def study_lasso(x,y,start,end,stepping):
    lasso_vector = np.arange(start,end,stepping)
    mean_squared_error_x = np.empty(lasso_vector.shape)
    betas = np.arange(10.0*lasso_vector.shape[0])
    betas = betas.reshape((10, lasso_vector.shape[0]))
    betas = np.zeros_like(betas)
    loo_score = np.empty(lasso_vector.shape[0] + 1)
    for i,alpha in enumerate(lasso_vector):
        reg = linear_model.Lasso(alpha)
        reg.fit(x, y)
        y_pred = reg.predict(x)
        beta = reg.coef_
        for j,beta_i in enumerate(beta):
            betas[j][i] = beta_i
        
        mean_squared_error_x[i] = mean_squared_error(y, y_pred)
        scores = cross_val_score(reg, X, Y, scoring='neg_mean_absolute_error',cv=cv, n_jobs=-1)
        loo_score[i] = np.mean(np.absolute(scores))
        #print(f"LOO score: {loo_score[i]} | MSE: {mean_squared_error_x[i]} for alpha = {alpha}")

        #comparing it to linear regression
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        Y_pred = regr.predict(x)
        scores = cross_val_score(regr, X, Y, scoring='neg_mean_absolute_error',cv=cv, n_jobs=-1)
        loo_score[-1] = np.mean(np.absolute(scores))

        #Matlab Data to be plotted 
        to_plot = {"lasso_vector": lasso_vector,"mean_squared_error_x": mean_squared_error_x, "label": "experiment"}
        savemat("to_plot_lasso_mse.mat",to_plot)
        to_plot = {"lasso_vector": lasso_vector,"betas": betas, "label": "experiment"}
        savemat("to_plot_lasso_betas.mat",to_plot)


        to_plot = {"lasso_vector": np.append(lasso_vector,"linear"),"loo_score": loo_score, "label": "experiment"}
        savemat("to_plot_loo_score.mat",to_plot)
                    
#Lasso
#study_lasso(X,Y,0.005,0.02, 0.001)  

#Linear
regr = linear_model.LinearRegression()
regr.fit(X, Y)
Y_pred = regr.predict(X)
print(f"Sum of squared errors {mean_squared_error(Y, Y_pred)} sklearn regression")






#Outliers
X = np.load('Xtrain_Regression2.npy')
Y = np.load('Ytrain_Regression2.npy')
X_test = np.load('Xtest_Regression2.npy')


X = np.load('Xtrain_Regression2.npy')
Y = np.load('Ytrain_Regression2.npy')

def remove_outliers(x, y):
    mse=10000
    while mse > 0.022:
        error=[]
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        y_pred = regr.predict(x)
        scores = cross_val_score(regr, x, y, scoring='neg_mean_absolute_error',cv=cv, n_jobs=-1)
        mse = mean_squared_error(y, y_pred)
        print(f"LOO score: {np.mean(np.absolute(scores))} | MSE: {mse} -> Y {y.shape} , X {x.shape}")
        for i,value in enumerate(x): error.append((y[i]-y_pred[i])**2)
        maxerror = max(error)
        maxerrori = error.index(maxerror)
        y=np.delete(y,maxerrori,axis=0)
        x=np.delete(x,maxerrori,axis=0)
    return x,y
X,Y = remove_outliers(X,Y)
print(Y.shape)
study_lasso(X,Y,0.05,0.1, 0.001)  