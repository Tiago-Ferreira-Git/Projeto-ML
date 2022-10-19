import numpy as np
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut,cross_val_score
from sklearn.metrics import mean_squared_error
cv = LeaveOneOut()
X = np.load('Xtrain_Regression2.npy')
Y = np.load('Ytrain_Regression2.npy')
X_test = np.load('Xtest_Regression2.npy')
def remove_outliers(x, y):
    maxerror=10000
    mse = 0
    while True:
        error=[]
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        y_pred = regr.predict(x)
        scores = cross_val_score(regr, x, y, scoring='neg_mean_absolute_error',cv=cv, n_jobs=-1)
        mse = mean_squared_error(y, y_pred)
        for i,value in enumerate(x): error.append((y[i]-y_pred[i])**2)
        maxerror = max(error)
        maxerrori = error.index(maxerror)
        if maxerror < 10*mse: break
        y=np.delete(y,maxerrori,axis=0)
        x=np.delete(x,maxerrori,axis=0)
        
    return x,y
X,Y = remove_outliers(X,Y)

reg = linear_model.Lasso(alpha = 0.006)
reg.fit(X, Y)
Y_test = reg.predict(X_test)

np.save("Ytest_Regression2", Y_test)
