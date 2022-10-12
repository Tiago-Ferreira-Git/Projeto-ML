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
# lasso_vector = np.arange(0.005,0.02, 0.001)
# mean_squared_error_x = np.empty(lasso_vector.shape)
# betas = np.arange(10.0*lasso_vector.shape[0])
# betas = betas.reshape((10, lasso_vector.shape[0]))
# betas = np.zeros_like(betas)
# loo_score = np.empty(lasso_vector.shape[0] + 1 )
# for i,alpha in enumerate(lasso_vector):
#     reg = linear_model.Lasso(alpha)
#     reg.fit(X, Y)
#     Y_pred = reg.predict(X)
#     beta = reg.coef_
#     for j,beta_i in enumerate(beta):
#         betas[j][i] = beta_i
#     
#     mean_squared_error_x[i] = mean_squared_error(Y, Y_pred)
#     scores = cross_val_score(reg, X, Y, scoring='neg_mean_absolute_error',cv=cv, n_jobs=-1)
#     loo_score[i] = np.mean(np.absolute(scores))
    #print(f"LOO score: {np.mean(np.absolute(scores))} | MSE: {mean_squared_error_x[i]} for alpha = {alpha}")

    
    
#Linear
regr = linear_model.LinearRegression()
regr.fit(X, Y)
Y_pred = regr.predict(X)
#print(f"Sum of squared errors {mean_squared_error(Y, Y_pred)} sklearn regression")


#polynomial




# scores = cross_val_score(regr, X, Y, scoring='neg_mean_absolute_error',cv=cv, n_jobs=-1)
# print(f"LeaveOneOut score for linear regression {np.mean(np.absolute(scores))}")
# loo_score[-1] = np.mean(np.absolute(scores))
# 
# to_plot = {"lasso_vector": lasso_vector,"mean_squared_error_x": mean_squared_error_x, "label": "experiment"}
# savemat("to_plot_lasso_mse.mat",to_plot)
# to_plot = {"lasso_vector": lasso_vector,"betas": betas, "label": "experiment"}
# savemat("to_plot_lasso_betas.mat",to_plot)
# 
# 
# to_plot = {"lasso_vector": np.append(lasso_vector,"linear"),"loo_score": loo_score, "label": "experiment"}
# savemat("to_plot_loo_score.mat",to_plot)





#Outliers
X = np.load('Xtrain_Regression2.npy')
Y = np.load('Ytrain_Regression2.npy')
X_test = np.load('Xtest_Regression2.npy')

delete_values = list()

erro_vector= np.arange(0.77,1.57, 0.01)

X = np.load('Xtrain_Regression2.npy')
Y = np.load('Ytrain_Regression2.npy')
maxerror=0
error=[]
for n in range(0,20):
    error=[]
    maxerror=0
    maxerrori=0
    reg = linear_model.Lasso(0.0017)
    reg.fit(X, Y)
    Y_pred = reg.predict(X)
    Y_test = reg.predict(X_test)
    scores = cross_val_score(reg, X, Y, scoring='neg_mean_absolute_error',cv=cv, n_jobs=-1)
    print(f"LOO score: {np.mean(np.absolute(scores))} | MSE: {mean_squared_error(Y, Y_pred)} -> Y {Y.shape} , X {X.shape}")
    for i,value in enumerate(Y):
        error.append((Y[i]-Y_pred[i])**2)
        #print(error[i])
        if maxerror < error[i]:
            maxerror=error[i]
            maxerrori=i
        else:
            continue
        pass
    print(maxerror)
    Y=np.delete(Y,maxerrori,axis=0)
    X=np.delete(X,maxerrori,0)
    pass
    
reg = linear_model.Lasso(0.0017)
reg.fit(X, Y)
Y_pred = reg.predict(X)
Y_test = reg.predict(X_test)
scores = cross_val_score(reg, X, Y, scoring='neg_mean_absolute_error',cv=cv, n_jobs=-1)
print(f"LOO score: {np.mean(np.absolute(scores))} | MSE: {mean_squared_error(Y, Y_pred)} -> Y {Y.shape} , X {X.shape}")


