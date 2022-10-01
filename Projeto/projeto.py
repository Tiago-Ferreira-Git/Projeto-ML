import numpy as np
from matplotlib import pyplot as plt
plt.close('all')
X = np.load('Xtrain_Regression1.npy')
Y = np.load('Ytrain_Regression1.npy')
#for i,column in enumerate(X.T):
#   X.T[i] = X.T[i] - np.mean(column)
#   
#   print(np.mean(column))
beta_aux = X.transpose().dot(X)

beta_aux = np.linalg.inv( beta_aux )

beta = (beta_aux.dot(X.transpose())).dot(Y)

print(f"Sum of squared errors {np.sum((Y-X.dot(beta))**2)}")
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption")
plt.figure(1) 
plt.plot(X,Y,"c.")
plt.plot(X,X.dot(beta),"b.") 
plt.show()
plt.pause(3)
plt.close('all')

# teste bro