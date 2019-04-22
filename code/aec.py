#_______________________________________________________________________________
# CE888 Project |        aec.py       | Ogulcan Ozer. | Feb. 2019 | UNFINISHED.
#_______________________________________________________________________________

##************************
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import utils as ku
from sklearn import cluster, datasets, metrics
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ae_config import ae_param as PAR
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from data import data_struct as DS
from ae import make_ae
import datetime

##************************
f = open('out.txt','a')
runtime = 'RUN: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(runtime, file = f)
np.random.seed(1)
path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))
results = [] 
#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def plot_decision_regions(data, label, classifier, resolution=0.02):
    h = .02 # point in the mesh [x_min, x_max]x[y_min, y_max].
    p =  'decision'
    im = str(data.shape[1]) + p
    # markers
    markers = ('s', 'x', 'o', '^', 'v', '.', ',','+', '<', '>', 's', 'd')
    
    
    # plot the decision surface
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.get_cmap("winter"))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(label)):
        plt.scatter(x=data[label == cl, 0], 
                    y=data[label == cl, 1],
                    alpha=0.6, 
                    c='k',
                    edgecolor='black',
                    marker='.',#markers[idx]
                    label=cl)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.title('Plot on the training data')
    plt.savefig('%s.png' % im , dpi=300)
    plt.show()

def read_data():

    #Read and seperate the mnist dataset.
    mnist = input_data.read_data_sets(os.path.join(path,'mnist'))
    tr = mnist.train.images
    trl = mnist.train.labels
    te = mnist.test.images
    tel = mnist.test.labels

    mnist_data = DS(tr,trl,te,tel,'mnist')

    #Read and seperate the HAR dataset.
    tr1 = pd.read_csv(os.path.join(path,'UCI HAR Dataset','train',
    'X_train.txt'),delim_whitespace=True,error_bad_lines=False, warn_bad_lines=False,
    header=None)
    trl1 = pd.read_csv(os.path.join(path,'UCI HAR Dataset','train',
    'y_train.txt'), error_bad_lines=False, warn_bad_lines=False,header=None)
    te1 = pd.read_csv(os.path.join(path,'UCI HAR Dataset','test',
    'X_test.txt'),delim_whitespace=True,error_bad_lines=False, warn_bad_lines=False,
    header=None)
    tel1 = pd.read_csv(os.path.join(path,'UCI HAR Dataset','test',
    'y_test.txt'),error_bad_lines=False, warn_bad_lines=False, header=None)
    har_data = DS(tr1,trl1.values.T[0],te1,tel1.values.T[0],'har')

    #Read and seperate the pulsar dataset.
    alldata = pd.read_csv(os.path.join(path,'HTRU2','HTRU_2.csv'),
    sep=',',error_bad_lines=False, warn_bad_lines=False,header=None)
    labels = pd.DataFrame(alldata[alldata.columns[8]])
    alldata.drop(alldata.columns[8], axis = 1, inplace = True)
    tr2, te2, trl2, tel2 = train_test_split(alldata, labels, test_size=0.20,
    random_state=42)
  
    pulsar_data = DS(tr2,trl2.values.T[0],te2,tel2.values.T[0],'pulsar')

    return mnist_data, har_data, pulsar_data

#Function for Kmeans clustering
def cluster_data(_data, cluster_size):
    
    data = _data.X_train
    labels = _data.Y_train
    # kmeans clustering 
    kM=cluster.KMeans(n_clusters = cluster_size, n_init=10)
    kM.fit(data)
    kM.labels_
    scores(labels,kM.labels_)
    return kM

def pca_comp(_data):
    p =  'pca'
    im = _data.name + p
    data = _data.X_train
    pca = PCA()
    reduced_train = pca.fit_transform(data)
    plt.bar(range(0, data.shape[1]), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.step(range(0, data.shape[1]), np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.savefig('%s.png' % im , dpi=300)
    plt.show()
    return reduced_train

def pca_linear(_data, n_components):
    train = _data.X_train
    test = _data.x_test
    name = 'pca_' + _data.name
    pca = PCA(n_components=n_components)
    reduced_train = pca.fit_transform(train)
    reduced_test = pca.transform(test)
    pca_data = DS(reduced_train,_data.Y_train,reduced_test,_data.y_test,name)
    return pca_data

def ae_build(_data,encoding,optimizer,loss):

    train = _data.X_train
    test = _data.x_test
    name = 'aelin_' + _data.name
    ae = make_ae(train.shape[1],[encoding],encoding,[train.shape[1]],[PAR.af_aenc],[PAR.af_adec])
    ae.autoencoder.compile(optimizer=optimizer, loss=loss)
    ae.autoencoder.fit(train, train,
                    epochs=PAR.ae_epoch,
                    batch_size=PAR.ae_batch,
                    shuffle=True,
                    validation_data=(test, test))

    # encoded_imgs = ae.encoder.predict(x_test)
    # decoded_imgs = ae.decoder.predict(encoded_imgs)
    reduced_train = ae.encoder.predict(train)
    reduced_test = ae.encoder.predict(test)
    ae_data = DS(reduced_train,_data.Y_train,reduced_test,_data.y_test,name)
    return ae_data, ae

def ae_softmax(_data,encoding,optimizer,loss):

    train = _data.X_train
    test = _data.x_test
    name = 'aesmx_' + _data.name
    ae = make_ae(train.shape[1],[encoding],encoding,[train.shape[1]],['softmax'],['linear'])
    ae.autoencoder.compile(optimizer=optimizer, loss=loss)
    ae.autoencoder.fit(train, train,
                    epochs=PAR.ae_epoch,
                    batch_size=PAR.ae_batch,
                    shuffle=True,
                    validation_data=(test, test))

    # encoded_imgs = ae.encoder.predict(x_test)
    # decoded_imgs = ae.decoder.predict(encoded_imgs)
    reduced_train = ae.encoder.predict(train)
    reduced_train = np.argmax(reduced_train)
    reduced_test = ae.encoder.predict(test)
    reduced_test = np.argmax(reduced_test)
    ae_data = DS(reduced_train,_data.Y_train,reduced_test,_data.y_test,name)
    return ae_data, ae

def scores(y_true, y_pred):
    hg, cp, v = metrics.homogeneity_completeness_v_measure(y_true, y_pred)
    a = metrics.accuracy_score(y_true, y_pred)
    print("Homogenity: " , hg )
    print("Completeness: " , cp )
    print("V-measure: " , v )
    print("Homogenity: " , hg, file = f)
    print("Completeness: " , cp, file = f)
    print("V-measure: ", file = f)
    print("Accuracy: " , a )
    print("Accuracy: " , a, file = f)

#-------------------------------------------------------------------------------
# Main program.
#-------------------------------------------------------------------------------

# Get all the data as a defined struct
d_mnist, d_har, d_pulsar = read_data()
d_har.Y_train[:] = [x - 1 for x in d_har.Y_train]
d_har.y_test[:] = [x - 1 for x in d_har.y_test]
# Print Shapes...
#
# for har,
print(d_har.X_train.shape)
print(d_har.Y_train.shape)
# for pulsar,
print(d_pulsar.X_train.shape)
print(d_pulsar.Y_train.shape)
# for mnist.
print(d_mnist.X_train.shape)
print(d_mnist.Y_train.shape)

print("Pulsar : ")
print("Pulsar : ", file = f)
cluster_data(d_pulsar, 2)
plot_decision_regions(data, label, classifier, 0.02):
print("Har : ")
print("Har : ", file = f)
cluster_data(d_har, 6)
plot_decision_regions(data, label, classifier, resolution=0.02):
print("Mnist : ")
print("Mnist : ", file = f)
cluster_data(d_mnist, 10)
plot_decision_regions(data, label, classifier, resolution=0.02):

#*** Clustering with linear PCA ***#
#check component variances#
pca_comp(d_pulsar)
pca_comp(d_har)
pca_comp(d_mnist)

p_pulsar = pca_linear(d_pulsar, 4)
p_har = pca_linear(d_har, 200)
p_mnist = pca_linear(d_mnist, 400)

print("Pulsar pca : ")
print("Pulsar pca : ", file = f)
cluster_data(p_pulsar, 2)
plot_decision_regions(data, label, classifier, resolution=0.02):
print("Har pca : ")
print("Har pca: ", file = f)
cluster_data(p_har, 6)
plot_decision_regions(data, label, classifier, resolution=0.02):
print("Mnist pca: ")
print("Mnist pca: ", file = f)
cluster_data(p_mnist, 10)
plot_decision_regions(data, label, classifier, resolution=0.02):
#*** Clustering using AE with linear activation functions (~PCA) and kmeans***#

ae_pulsar, ae_p = ae_build(d_pulsar, 4, PAR.ae_opt, PAR.ae_loss)
ae_har, ae_h = ae_build(d_har, 200, PAR.ae_opt, PAR.ae_loss)
ae_mnist, ae_m = ae_build(d_mnist, 400, PAR.ae_opt, PAR.ae_loss)

print("Pulsar ae : ")
print("Pulsar ae : ", file = f)
cluster_data(ae_pulsar, 2)
plot_decision_regions(data, label, classifier, resolution=0.02):
print("Har ae : ")
print("Har ae: ", file = f)
cluster_data(ae_har, 6)
plot_decision_regions(data, label, classifier, resolution=0.02):
print("Mnist ae: ")
print("Mnist ae: ", file = f)
cluster_data(ae_mnist, 10)
plot_decision_regions(data, label, classifier, resolution=0.02):

sf_pulsar, sf_p = ae_softmax(d_pulsar, 8, PAR.ae_opt, PAR.ae_loss)
sf_har, sf_h = ae_softmax(d_har, 6, PAR.ae_opt, PAR.ae_loss)
sf_mnist, sf_m = ae_softmax(d_mnist, 10, PAR.ae_opt, PAR.ae_loss)

print("Pulsar ae_softmax : ")
print("Pulsar ae_softmax : ", file = f)
scores(sf_pulsar.X_train,sf_pulsar.Y_train)

print("Har ae_softmax : ")
print("Har ae_softmax: ", file = f)
scores(sf_har.X_train,sf_har.Y_train)

print("Mnist ae_softmax: ")
print("Mnist ae_softmax: ", file = f)
scores(sf_mnist.X_train,sf_mnist.Y_train)





#-------------------------------------------------------------------------------
#                               End of aec.py
#-------------------------------------------------------------------------------
