##************************
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential,Model
from keras.layers import Dense, Input
from sklearn import datasets
from sklearn import cluster, datasets, metrics
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn
from keras import utils as ku
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ae import make_ae
from data import data_struct as DS
##************************
def plot_decision_regions(X, y, classifier, resolution=0.02):
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v', '.', ',','+', '<', '>', 's', 'd')
    
    
    # plot the decision surface
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.get_cmap("winter"))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c='k',
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.title('Plot on the training data')
    plt.show()
#Read and seperate the mnist dataset.
#from keras.datasets import mnist
import numpy as np

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print (x_train.shape)
# print (x_test.shape)
# Define custom loss

digits = load_digits()
data = scale(digits.data.astype(np.float32))

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target.astype(np.float32)
print(labels.shape)
print(data.shape)

def custom_loss(y_true ,y_pred):
    y_pred = K.argmax(y_pred)
    y_pred = K.cast(y_pred, dtype='float32')
    ypred_num = K.eval(y_pred)
    return sklearn.metrics.silhouette_score(y_true, ypred_num, metric='euclidean')


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
one_hot_train = ku.to_categorical(y_train, num_classes=10)
print(one_hot_train)
ae = make_ae(64,[10],10,[64],['softmax'],['relu'])



ae.autoencoder.compile(optimizer='sgd', loss=['kullback_leibler_divergence','binary_crossentropy'],
                loss_weights=[0.2, 1],
                metrics=['accuracy'])
ae.autoencoder.fit(x_train, [one_hot_train, x_train],
                epochs=200,
                batch_size=64,
                shuffle=True)

reduced_data = ae.encoder.predict(data)
#decoded_imgs = ae.decoder.predict(encoded_imgs)
results = []
for i in range (reduced_data.shape[0]):
    results.append(np.argmax(reduced_data[i]))


print('all data')
print(reduced_data)
print(reduced_data.shape)
a = metrics.accuracy_score(labels, np.argmax(reduced_data, axis=1))
print('Accuracy: ', a )
print(labels)
print(results)




# kmeans = cluster.KMeans(init='k-means++', n_clusters=10, n_init=10)
# kmeans.fit(reduced_data)
# hg, cp, v = metrics.homogeneity_completeness_v_measure(labels, kmeans.labels_)
# print("Homogenity: " , hg )
# print("Completeness: " , cp )
# print("V-measure: " , v )
# plot_decision_regions(reduced_data[:,:2],labels,kmeans)


# print (encoded_imgs.shape)
# print (decoded_imgs.shape)
# # use Matplotlib (don't ask)
# import matplotlib.pyplot as plt

# n = 10  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(8,8))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(8, 8))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

