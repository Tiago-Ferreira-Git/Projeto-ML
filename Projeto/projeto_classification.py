import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
X = np.load('numpy_files/Xtrain_Classification1.npy')
print(X.shape)
X = X.reshape((8273,30, 30,3))
print(len(X))
# for image in X:
#     plt.imshow(image)
#     plt.show()
